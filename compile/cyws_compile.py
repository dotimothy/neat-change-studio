#!/usr/bin/env python3
"""Quantize and compile CYWS (two image inputs) for Modalix with the SiMa Model Compiler.

Run inside the Model Compiler environment (compile/compile.sh does this):

    python3 compile/cyws_compile.py --precision bf16 --out build/bf16
    python3 compile/cyws_compile.py --precision int8 --calib build/calib/calib_pairs.npz \
        --calib-method min_max --out build/int8_minmax

Also used for the probe sub-graphs (compile/probes/), which is why the input
names and shapes are read from the ONNX file instead of being hard-coded.

Precisions:
  bf16   bf16 weights and activations (no real calibration needed; the accurate build)
  int8   int8 weights and activations, calibrated on paired before/after samples
  w8a16  int8 weights, bf16 activations

Calibration (--calib) is an .npz holding one array per ONNX input, keyed by the
input name, each (N, C, H, W) or (N, H, W, C) float32 in the model's input
range. Row i of every array belongs to the same before/after pair: the stock
quantize_compile.py feeds one image set to every input, which for a change
detector means calibrating on "no change" only.

Afterwards the script checks the MPK: exactly one MLA ELF and no A65 library
(pyneat refuses split MPKs). It exits non-zero otherwise, after printing the
nodes the compiler could not place.
"""
import argparse
import json
import logging
import os
import sys
import tarfile
import time

import numpy as np
import onnx

from afe.apis.defines import (
    default_quantization, quantization_scheme, bfloat16_scheme,
    RequantizationMode, CalibrationMethod, gen2_target, InputName,
)
from afe.apis.loaded_net import load_model
from afe.core.utils import convert_data_generator_to_iterable
from afe.ir.tensor_type import ScalarType
from afe.load.importers.general_importer import ImporterParams, ModelFormat
from sima_utils.data.data_generator import DataGenerator


def log(msg=""):
    print(msg, flush=True)


def onnx_io(path):
    m = onnx.load(path, load_external_data=False)
    inits = {i.name for i in m.graph.initializer}
    ins = [(i.name, tuple(d.dim_value for d in i.type.tensor_type.shape.dim))
           for i in m.graph.input if i.name not in inits]
    outs = [o.name for o in m.graph.output]
    return ins, outs


def calibration_set(path, inputs, n, precision):
    """Paired calibration samples as NHWC arrays, one dict entry per input."""
    data = {}
    if path and os.path.exists(path):
        z = np.load(path)
        for name, shape in inputs:
            if name not in z:
                sys.exit(f"{path}: missing array '{name}' (has {list(z.keys())})")
            a = z[name].astype(np.float32)[:n]
            if a.shape[1:] == shape[1:]:                       # NCHW -> NHWC
                a = a.transpose(0, 2, 3, 1)
            elif a.shape[1:] != (shape[2], shape[3], shape[1]):
                sys.exit(f"{path}:{name}: expected (N,{shape[1:]}) or NHWC, got {a.shape}")
            data[name] = np.ascontiguousarray(a)
        lens = {len(a) for a in data.values()}
        if len(lens) != 1:
            sys.exit(f"{path}: inputs have different sample counts {lens}")
        log(f"  calibration: {lens.pop()} paired samples from {path}")
    else:
        if precision != "bf16":
            sys.exit(f"--calib {path!r} not found; int8/w8a16 need real paired calibration data")
        rng = np.random.default_rng(0)
        for name, shape in inputs:
            data[name] = rng.random((2, shape[2], shape[3], shape[1]), dtype=np.float32)
        log("  calibration: 2 random samples (bf16 has no scales to fit)")
    return convert_data_generator_to_iterable(DataGenerator({InputName(k): v for k, v in data.items()}))


def summarize_mpk(out_dir):
    """Count MLA ELFs, A65 libraries and plugins per processor; read the MPK input order."""
    mpks = [f for f in os.listdir(out_dir) if f.endswith("_mpk.tar.gz")]
    if not mpks:
        log("  no *_mpk.tar.gz produced")
        return None
    path = os.path.join(out_dir, mpks[0])
    with tarfile.open(path) as t:
        names = t.getnames()
        manifest = next((n for n in names if n.endswith("_mpk.json")), None)
        mpk = json.load(t.extractfile(manifest)) if manifest else {}
    plugins = mpk.get("plugins", [])
    elfs = [n for n in names if n.endswith("_mla.elf")]
    a65 = [n for n in names if n.endswith("_a65.so")]
    procs = {}
    for p in plugins:
        procs[p["processor"]] = procs.get(p["processor"], 0) + 1
    log(f"  MPK: {path} ({os.path.getsize(path) / 2**20:.0f} MB)")
    log(f"  MLA ELFs: {len(elfs)} {elfs}")
    log(f"  A65 libraries: {len(a65)}")
    log(f"  plugins by processor: {procs}")
    return {"mpk": path, "mla_elfs": elfs, "a65": a65, "plugins": procs,
            "input_nodes": mpk.get("input_nodes"), "output_nodes": mpk.get("output_nodes")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="build/onnx/cyws_fused.onnx")
    ap.add_argument("--precision", choices=["bf16", "int8", "w8a16"], default="bf16")
    ap.add_argument("--calib", default="build/calib/calib_pairs.npz")
    ap.add_argument("--num-calib", type=int, default=96)
    ap.add_argument("--calib-method", default="mse", help="mse, min_max, moving_average, entropy, ...")
    ap.add_argument("--requant-mode", choices=["sima", "tflite"], default="sima")
    ap.add_argument("--smooth-quant", action="store_true")
    ap.add_argument("--channel-eq", action="store_true")
    ap.add_argument("--bias-correction", action="store_true")
    ap.add_argument("--name", default=None, help="model name inside the MPK (default cyws_<precision>)")
    ap.add_argument("--out", default=None, help="output directory (default build/<precision>)")
    ap.add_argument("--no-compile", action="store_true", help="quantize only")
    args = ap.parse_args()

    name = args.name or f"cyws_{args.precision}"
    out = os.path.abspath(args.out or os.path.join("build", args.precision))
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    inputs, outputs = onnx_io(args.model)
    log(f"[1/3] load {args.model}")
    log(f"  inputs {inputs}  outputs {outputs}")
    net = load_model(ImporterParams(
        format=ModelFormat.onnx, file_paths=[args.model], input_names=[n for n, _ in inputs],
        input_shapes=[s for _, s in inputs], input_types=[ScalarType.float32] * len(inputs),
        layout="NCHW", output_names=outputs), target=gen2_target)

    log(f"[2/3] quantize: {args.precision}")
    act = bfloat16_scheme() if args.precision in ("bf16", "w8a16") else quantization_scheme(True, False, 8)
    wgt = bfloat16_scheme() if args.precision == "bf16" else quantization_scheme(False, True, 8)
    rq = RequantizationMode.sima if args.requant_mode == "sima" else RequantizationMode.tflite
    cfg = (default_quantization.with_activation_quantization(act).with_weight_quantization(wgt)
           .with_requantization_mode(rq).with_calibration(CalibrationMethod.from_str(args.calib_method)))
    if args.smooth_quant:
        cfg = cfg.with_smooth_quant(True)
    if args.channel_eq:
        cfg = cfg.with_channel_equalization(True)
    if args.bias_correction:
        cfg = cfg.with_bias_correction(True)
    calib = calibration_set(args.calib if args.precision != "bf16" else None, inputs,
                            args.num_calib, args.precision)
    q = net.quantize(calibration_data=calib, quantization_config=cfg, model_name=name, log_level=logging.INFO)
    q.save(model_name=name, output_directory=out)
    log(f"  quantized model saved to {out} ({time.time() - t0:.0f} s)")

    summary = {"model": args.model, "precision": args.precision, "name": name,
               "calib_method": args.calib_method if args.precision != "bf16" else None,
               "calib": args.calib if args.precision != "bf16" else None,
               "smooth_quant": args.smooth_quant, "channel_eq": args.channel_eq,
               "bias_correction": args.bias_correction, "onnx_inputs": [n for n, _ in inputs]}
    ok = True
    if not args.no_compile:
        log("[3/3] compile for Modalix")
        q.compile(output_path=out, batch_size=1, log_level=logging.INFO)
        log(f"  compiled ({time.time() - t0:.0f} s total)")
        s = summarize_mpk(out) or {}
        summary.update(s)
        ok = len(s.get("mla_elfs", [])) == 1 and not s.get("a65")
        summary["single_mla_elf"] = ok
        log(f"  single MLA ELF, no A65: {'YES' if ok else 'NO'}")
    summary["seconds"] = round(time.time() - t0)
    with open(os.path.join(out, "compile_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
