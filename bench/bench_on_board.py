#!/usr/bin/env python3
"""Benchmark the compiled CYWS model on the Modalix DevKit.

Two measurements on the same MPK:
  1. Streamed throughput through the app's own NeatBackend: N random
     before/after pairs pushed with up to --in-flight outstanding, which is how
     the app feeds tiles. Reports wall-clock ms per pair and per-inference
     latency percentiles (push -> output in host memory).
  2. pyneat Model.benchmark (NEAT's synthetic benchmark, plugin latency
     included): latency, FPS and power where the runtime reports them.
     It builds its own pipeline, so it runs after the stream is closed.

  python3 bench/bench_on_board.py --model-path models/cyws_bf16_mpk.tar.gz
  python3 bench/bench_on_board.py --model-path ... --stream 300 --num-samples 200 --json out/bench.json

Stop any other MLA app first (one MLA user at a time).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for cand in (HERE.parent / "app", HERE / "app"):
    if (cand / "backends.py").exists():
        sys.path.insert(0, str(cand))
        break

import backends  # noqa: E402
from common import MODEL_SIZE  # noqa: E402


def stream_bench(be: backends.Backend, n: int, warmup: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    pool = [rng.random((MODEL_SIZE, MODEL_SIZE, 3), dtype=np.float32) for _ in range(8)]

    def items(count: int):
        for i in range(count):
            yield i, pool[i % len(pool)], pool[(i + 3) % len(pool)]

    for _ in be.infer_stream(items(warmup)):
        pass
    t0 = time.perf_counter()
    got = sum(1 for _ in be.infer_stream(items(n)))
    wall = time.perf_counter() - t0
    if got != n:
        raise RuntimeError(f"expected {n} outputs, got {got}")
    st = be.stats()
    return {
        "pairs": n,
        "in_flight": be.max_in_flight,
        "wall_s": round(wall, 3),
        "ms_per_pair": round(wall * 1e3 / n, 2),
        "pairs_per_s": round(n / wall, 2),
        "latency_ms": st.get("latency"),  # includes the warmup inferences
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True, help="compiled *_mpk.tar.gz")
    ap.add_argument("--stream", type=int, default=200, help="pairs for the streamed run (0 = skip)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--in-flight", type=int, default=2, help="max outstanding pushes, as in the app")
    ap.add_argument("--num-samples", type=int, default=200, help="Model.benchmark samples (0 = skip)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="also write the results here")
    a = ap.parse_args()

    res: dict = {"model": str(a.model_path)}
    be = backends.make_backend("neat", a.model_path, max_in_flight=a.in_flight)
    try:
        if a.stream > 0:
            res["stream"] = stream_bench(be, a.stream, a.warmup, a.seed)
            s = res["stream"]
            lat = s["latency_ms"] or {}
            print(f"stream: {s['pairs']} pairs, {s['ms_per_pair']} ms/pair ({s['pairs_per_s']} pairs/s), "
                  f"in flight {s['in_flight']}; latency p50 {lat.get('p50_ms')} p90 {lat.get('p90_ms')} "
                  f"max {lat.get('max_ms')} ms")
        if a.num_samples > 0:
            try:
                res["benchmark"] = be.benchmark(a.num_samples)  # releases the runner; must come last
                b = res["benchmark"]
                print("Model.benchmark: " + ", ".join(f"{k}={round(v, 3) if isinstance(v, float) else v}"
                                                      for k, v in b.items()))
            except Exception as e:  # benchmark support varies by runtime build
                res["benchmark_error"] = f"{type(e).__name__}: {e}"
                print(f"Model.benchmark unavailable: {res['benchmark_error']}")
    finally:
        be.close()
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1))
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
