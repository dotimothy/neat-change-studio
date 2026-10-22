#!/usr/bin/env python3
"""Run the fixed evaluation tensors (build/calib/eval_inputs.npz) through one backend.

The same script runs everywhere, through the app's own backend classes, so the
numbers compare like for like:
  host  : --backend onnx --model build/onnx/cyws_fused.onnx         (FP32 reference)
  SDK   : --backend afe  --model build/bf16 --name cyws_bf16        (compiler host simulation)
  board : --backend neat --model models/cyws_bf16_mpk.tar.gz         (pyneat on the MLA)

Writes <out>.npy: float16 (N, 10, 256, 256) canonical outputs, and <out>.json
with timings, input_specs (neat) and Model.benchmark (neat, --benchmark).

Usage: run_model.py --backend onnx --model build/onnx/cyws_fused.onnx --out build/eval/fp32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for cand in (HERE.parent / "app", HERE.parent):           # repo layout, or board layout (app/ copied flat)
    if (cand / "backends.py").exists():
        sys.path.insert(0, str(cand))
        break
import backends  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["onnx", "afe", "neat"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default="cyws_bf16", help="afe: model name inside --model dir")
    ap.add_argument("--inputs", default=str(HERE.parent / "build/calib/eval_inputs.npz"))
    ap.add_argument("--rows", default=None, help="start:end subset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--benchmark", type=int, default=0, help="neat: Model.benchmark num_samples (0 = skip)")
    ap.add_argument("--no-graph", action="store_true", help="neat: use ModelRunner instead of Graph->Run")
    args = ap.parse_args()

    z = np.load(args.inputs)
    xb, xa, names = z["in0_before"], z["in1_after"], z["names"]
    if args.rows:
        a, b = (int(v) for v in args.rows.split(":"))
        xb, xa, names = xb[a:b], xa[a:b], names[a:b]
    hwc = lambda x: np.ascontiguousarray(x.transpose(1, 2, 0), np.float32)

    if args.backend == "onnx":
        be = backends.OnnxBackend(args.model)
    elif args.backend == "afe":
        be = backends.AfeBackend(args.name, args.model)
    else:
        be = backends.NeatBackend(args.model, use_graph=not args.no_graph)
    info: dict = {"backend": args.backend, "model": args.model, "n": int(len(names))}
    try:
        if args.backend == "neat":
            info["describe"] = be.stats()
        outs = np.zeros((len(names), 10, 256, 256), np.float16)
        t0 = time.perf_counter()
        items = ((i, hwc(xb[i]), hwc(xa[i])) for i in range(len(names)))
        for i, out in be.infer_stream(items):
            outs[i] = out.astype(np.float16)
            if args.backend == "afe":
                print(f"  {i + 1}/{len(names)}  {time.perf_counter() - t0:.0f} s", flush=True)
        dt = time.perf_counter() - t0
        info["seconds"] = round(dt, 3)
        info["ms_per_pair_streamed"] = round(1000 * dt / max(1, len(names)), 2)
        info["stats"] = be.stats()
        if args.backend == "neat" and args.benchmark:
            info["benchmark"] = be.benchmark(num_samples=args.benchmark)
    finally:
        be.close()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out + ".npy", outs)
    info["names"] = [str(n) for n in names]
    Path(args.out + ".json").write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps({k: v for k, v in info.items() if k != "names"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
