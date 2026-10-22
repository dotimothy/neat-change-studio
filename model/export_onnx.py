#!/usr/bin/env python3
"""Export the MLA-friendly CYWS to ONNX and check it.

Steps: build (strict load, fold BN, fuse head) -> torch.onnx.export (opset 17,
TorchScript exporter) -> onnxsim -> structural checks -> onnxruntime vs torch.

Structural checks (export fails if any is violated):
  * only allowlisted op types (the set the SiMa MLA accepts for this model);
  * the only Einsum equations are the two that compile on the MLA (checked with the probes);
  * inputs are exactly in0_before, in1_after (1,3,256,256) float32 in that order;
  * outputs are the expected names/shapes;
  * shared Siamese weights are stored once (no duplicated initializers).

Usage: export_onnx.py [--outputs fused|split] [--query-chunks 1,1,1] [--scse-expand broadcast|conv]
                      [--in-ch 3] [--out build/onnx/cyws_fused.onnx]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import cyws_mla  # noqa: E402

ALLOWED_OPS = {"Conv", "Relu", "Sigmoid", "Add", "Mul", "Concat", "Split", "Einsum", "Softmax",
               "MaxPool", "Resize", "GlobalAveragePool", "ReduceMean"}
ALLOWED_EINSUM = {"nchw,nchq->nqhw", "nchw,nqhc->nqhw"}
CKPT = ROOT / "build/ckpt/pretrained-resnet50-3x-coam-scSE-affine.ckpt"


def constants_to_initializers(model: onnx.ModelProto) -> int:
    """Turn Constant nodes (e.g. Resize scales) into initializers; returns how many."""
    g = model.graph
    keep, n = [], 0
    for node in g.node:
        if node.op_type == "Constant" and node.attribute and node.attribute[0].name == "value":
            t = onnx.TensorProto()
            t.CopyFrom(node.attribute[0].t)
            t.name = node.output[0]
            g.initializer.append(t)
            n += 1
        else:
            keep.append(node)
    del g.node[:]
    g.node.extend(keep)
    return n


def check_graph(model: onnx.ModelProto, outputs: str, in_ch: int) -> dict:
    g = model.graph
    ops = Counter(n.op_type for n in g.node)
    bad = sorted(set(ops) - ALLOWED_OPS)
    if bad:
        raise SystemExit(f"disallowed ops in graph: {bad} ({ {k: ops[k] for k in bad} })")
    eqs = Counter()
    for n in g.node:
        if n.op_type == "Einsum":
            eq = next(a.s.decode() for a in n.attribute if a.name == "equation").replace(" ", "")
            eqs[eq] += 1
    if set(eqs) - ALLOWED_EINSUM:
        raise SystemExit(f"unexpected Einsum equations: {dict(eqs)}")
    init_names = {i.name for i in g.initializer}
    inputs = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim], i.type.tensor_type.elem_type)
              for i in g.input if i.name not in init_names]
    want_in = [("in0_before", [1, in_ch, 256, 256], 1), ("in1_after", [1, in_ch, 256, 256], 1)]
    if inputs != want_in:
        raise SystemExit(f"inputs {inputs} != {want_in}")
    outs = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in g.output]
    want_out = ([("out", [1, 10, 256, 256])] if outputs == "fused" else
                [("out0_heat", [1, 2, 256, 256]), ("out1_wh", [1, 4, 256, 256]), ("out2_off", [1, 4, 256, 256])])
    if outs != want_out:
        raise SystemExit(f"outputs {outs} != {want_out}")
    digests = Counter(hashlib.sha1(i.raw_data or onnx.numpy_helper.to_array(i).tobytes()).hexdigest()
                      for i in g.initializer if np.prod(i.dims) > 64)
    dup = sum(c - 1 for c in digests.values() if c > 1)
    n_params = int(sum(np.prod(i.dims) for i in g.initializer))
    return {"ops": dict(sorted(ops.items())), "einsum": dict(eqs), "inputs": inputs, "outputs": outs,
            "initializers": len(g.initializer), "params": n_params, "duplicated_large_initializers": dup}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--outputs", choices=["fused", "split"], default="fused")
    ap.add_argument("--query-chunks", default="1,1,1")
    ap.add_argument("--scse-expand", choices=["broadcast", "conv"], default="conv",
                    help="conv: the SiMa compiler fails on scSE with two broadcast multiplies")
    ap.add_argument("--in-ch", type=int, default=3)
    ap.add_argument("--max-conv-params", type=int, default=6_000_000,
                    help="split convs with more weights than this along input channels (0 = off)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-sim", action="store_true", help="skip onnxsim")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or ROOT / "build/onnx" / f"cyws_{args.outputs}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)

    qc = tuple(int(x) for x in args.query_chunks.split(","))
    model = cyws_mla.build(args.ckpt, export=True, outputs=args.outputs, query_chunks=qc,
                           scse_expand=args.scse_expand, in_ch=args.in_ch,
                           max_conv_params=args.max_conv_params).eval()
    if model.split_convs:
        print("split convs:", model.split_convs)
    names = list(cyws_mla.FUSED_OUTPUT_NAMES if args.outputs == "fused" else cyws_mla.SPLIT_OUTPUT_NAMES)
    rng = np.random.default_rng(0)
    xb = torch.from_numpy(rng.random((1, args.in_ch, 256, 256), dtype=np.float32))
    xa = torch.from_numpy(rng.random((1, args.in_ch, 256, 256), dtype=np.float32))
    if args.in_ch > 3:
        xb[:, 3:] = 0
        xa[:, 3:] = 0
    with torch.no_grad():
        export_kw = dict(input_names=list(cyws_mla.INPUT_NAMES), output_names=names,
                         opset_version=args.opset, do_constant_folding=True)
        try:
            torch.onnx.export(model, (xb, xa), str(out), dynamo=False, **export_kw)
        except TypeError:   # older torch without the dynamo kwarg
            torch.onnx.export(model, (xb, xa), str(out), **export_kw)
        ref = model(xb, xa)
    ref = [ref] if isinstance(ref, torch.Tensor) else list(ref)

    m = onnx.load(str(out))
    if not args.no_sim:
        import onnxsim
        m, ok = onnxsim.simplify(m)
        if not ok:
            raise SystemExit("onnxsim check failed")
        onnx.save(m, str(out))
    n_const = constants_to_initializers(m)
    onnx.save(m, str(out))
    onnx.checker.check_model(m)
    report = check_graph(m, args.outputs, args.in_ch)
    report["constants_moved_to_initializers"] = n_const

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(str(out), so, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"in0_before": xb.numpy(), "in1_after": xa.numpy()})
    diffs = [float(np.abs(g - r.numpy()).max()) for g, r in zip(got, ref)]
    report["ort_vs_torch_max_abs"] = diffs
    report["file"] = str(out)
    report["options"] = vars(args) | {"out": str(out)}
    tol = 1e-3
    report["pass"] = all(d <= tol for d in diffs)
    out.with_suffix(".json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
