#!/usr/bin/env python3
"""Build paired model inputs from the testCYWS pairs, through the app's own preprocessing.

Writes two .npz files, each holding in0_before / in1_after as (N, 3, 256, 256)
float32 (RGB / 255) plus a `names` array, row i of both inputs being one pair:

  build/calib/calib_pairs.npz   calibration for int8 / w8a16 compiles: the global
                                view of every pair plus level-1 tiles (tiles over
                                ground-truth changes first, then random ones), some
                                after registration, so the activation ranges cover
                                what the app feeds.
  build/calib/eval_inputs.npz   fixed tensor set for numeric comparisons (ONNX FP32
                                vs afe vs board): all globals, a set of tiles and
                                A/B pairs whose before/after differ strongly (to
                                detect a swapped input order on the board).

Both go through app/ingestion.crop_pair, app/tiling.plan_windows and (optionally)
app/registration, so the compiled model is checked on exactly what the app sends.

Usage: make_calib.py [--data DIR] [--calib-tiles 68] [--eval-tiles 64]

--data defaults to $CYWS_DATA, or data/testCYWS in the repo.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = Path(os.environ.get("CYWS_DATA") or ROOT / "data" / "testCYWS")
sys.path.insert(0, str(ROOT / "app"))
from dataset_loader import discover_pairs  # noqa: E402
from ingestion import crop_pair, load_pair  # noqa: E402
from registration import register, warp_before  # noqa: E402
from tiling import plan_windows  # noqa: E402


def nchw(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA, type=Path)
    ap.add_argument("--out", default=ROOT / "build/calib", type=Path)
    ap.add_argument("--calib-tiles", type=int, default=68)
    ap.add_argument("--eval-tiles", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    items, warnings = discover_pairs(args.data)
    for w in warnings:
        print("warning:", w)
    globals_, tiles_change, tiles_other = [], [], []
    for k, item in enumerate(items):
        pair = load_pair(item)
        before = pair.before
        if k % 3 == 0:   # every third pair also contributes registered tiles
            reg = register(pair.before, pair.after)
            if reg.ok:
                before = warp_before(pair.before, reg, pair.after)
        H, W = pair.after.shape[:2]
        wins = plan_windows(H, W, levels=(0, 1))
        for win in wins:
            b, a = crop_pair(before, pair.after, win)
            name = f"{pair.pair_id}/L{win.level}w{win.id}"
            if win.level == 0:
                globals_.append((name, b, a))
                continue
            changed = pair.gt_mask is not None and pair.gt_mask[win.slices()].mean() > 0.002
            (tiles_change if changed else tiles_other).append((name, b, a))
        print(f"{pair.pair_id}: {len(wins)} windows")

    def pick(pool, n):
        idx = rng.permutation(len(pool))[:n]
        return [pool[i] for i in sorted(idx)]

    def save(path: Path, rows) -> None:
        np.savez_compressed(path, in0_before=np.stack([nchw(b) for _, b, _ in rows]),
                            in1_after=np.stack([nchw(a) for _, _, a in rows]),
                            names=np.array([n for n, _, _ in rows]))
        print(f"wrote {path}: {len(rows)} pairs")

    n_change = min(len(tiles_change), args.calib_tiles // 2)
    calib = globals_ + pick(tiles_change, n_change) + pick(tiles_other, args.calib_tiles - n_change)
    save(args.out / "calib_pairs.npz", calib)

    eval_rows = list(globals_)
    eval_rows += pick(tiles_change, min(len(tiles_change), args.eval_tiles // 2))
    eval_rows += pick(tiles_other, args.eval_tiles - min(len(tiles_change), args.eval_tiles // 2))
    # A/B swap probes: before and after from *different* scenes, so the two
    # branches' outputs differ a lot and a swapped input order is unmistakable.
    for i in range(3):
        n1, b1, _ = globals_[i]
        n2, _, a2 = globals_[(i + len(globals_) // 2) % len(globals_)]
        eval_rows.append((f"swap{i}:{n1.split('/')[0]}|{n2.split('/')[0]}", b1, a2))
    save(args.out / "eval_inputs.npz", eval_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
