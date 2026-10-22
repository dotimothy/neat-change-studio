#!/usr/bin/env python3
"""Compare canonical CYWS outputs from two backends on the same eval tensors.

Metrics per backend pair (test vs reference, usually bf16 vs FP32):
  heat_cosine        cosine similarity of the two heat channels, per sample, on samples
                     whose reference heat peak is >= thr/2 (below that the heatmap holds
                     no detection and its cosine is noise)
  heat_max_abs       largest |test - ref| heat value over all samples
  wh_off_rel_err     median relative error of wh/off at reference peaks (score >= thr)
  box_agreement      fraction of reference boxes (score >= thr) matched by a test box
                     (score >= thr/2) with IoU >= 0.7
  ap50               VOC AP at IoU 0.5 of test boxes, using reference boxes as ground truth
  swap check         for rows named swap*: the test output must be much closer to the
                     reference for (A, B) than to the reference with the branches swapped

Pass criteria (bf16): heat cosine >= 0.99 (min over samples with a detectable
peak), heat max abs <= 0.1, box agreement >= 0.95, AP50 >= 0.90, swap check passes.

Usage: compare.py build/eval/fp32 build/eval/board_bf16 [--thr 0.3]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
from post_process import decode_centernet, iou_matrix  # noqa: E402


def load(prefix: str):
    out = np.load(prefix + ".npy").astype(np.float32)
    meta = json.loads(Path(prefix + ".json").read_text())
    return out, meta["names"]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 1.0


def voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def swap_branches(o: np.ndarray) -> np.ndarray:
    """Canonical output with the before/after branches exchanged."""
    perm = [1, 0, 4, 5, 2, 3, 8, 9, 6, 7]
    return o[perm]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--thr", type=float, default=0.3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ref, names = load(args.ref)
    test, tnames = load(args.test)
    if names != tnames:
        sys.exit("reference and test were run on different inputs")

    cos, rel, agree_n, agree_d = [], [], 0, 0
    heat_abs = 0.0
    dets, n_gt = [], 0
    swap = []
    for i, name in enumerate(names):
        r, t = ref[i], test[i]
        heat_abs = max(heat_abs, float(np.abs(r[:2] - t[:2]).max()))
        if r[:2].max() >= args.thr / 2:   # cosine of a heatmap with no detection is noise
            cos.append(cosine(r[:2], t[:2]))
        for br in ("before", "after"):
            rb = decode_centernet(r, br, thr=args.thr)
            tb = decode_centernet(t, br, thr=args.thr / 2)
            n_gt += len(rb)
            if len(rb):
                agree_d += len(rb)
                if len(tb):
                    agree_n += int((iou_matrix(rb[:, :4], tb[:, :4]).max(1) >= 0.7).sum())
                # wh/off relative error at reference peaks
                c = {"before": (0, 2, 6), "after": (1, 4, 8)}[br]
                for x1, y1, x2, y2, _ in rb:
                    cx, cy = int(np.clip((x1 + x2) / 2, 0, 255)), int(np.clip((y1 + y2) / 2, 0, 255))
                    rv = r[c[1]:c[1] + 2, cy, cx]
                    tv = t[c[1]:c[1] + 2, cy, cx]
                    rel.append(float(np.abs(tv - rv).max() / max(1.0, np.abs(rv).max())))
            # AP50 with the reference boxes as ground truth
            tb_all = decode_centernet(t, br, thr=0.05)
            used = np.zeros(len(rb), bool)
            for b in tb_all:
                if len(rb):
                    ious = iou_matrix(b[None, :4], rb[:, :4])[0]
                    j = int(ious.argmax())
                    hit = ious[j] >= 0.5 and not used[j]
                    used[j] |= hit
                else:
                    hit = False
                dets.append((float(b[4]), hit))
        if name.startswith("swap"):
            d_same = float(np.abs(t - r).mean())
            d_swap = float(np.abs(t - swap_branches(r)).mean())
            swap.append({"name": name, "dist_same": d_same, "dist_swapped": d_swap,
                         "ok": d_same < 0.5 * d_swap})

    dets.sort(key=lambda d: -d[0])
    tp = np.cumsum([d[1] for d in dets]) if dets else np.zeros(0)
    fp = np.cumsum([not d[1] for d in dets]) if dets else np.zeros(0)
    ap50 = voc_ap(tp / max(1, n_gt), tp / np.maximum(1, tp + fp)) if n_gt else 1.0
    res = {
        "n": len(names),
        "heat_cosine_min": min(cos) if cos else None,
        "heat_cosine_mean": float(np.mean(cos)) if cos else None,
        "heat_cosine_samples": len(cos),
        "heat_max_abs": heat_abs,
        "wh_off_rel_err_median": float(np.median(rel)) if rel else None,
        "wh_off_rel_err_p95": float(np.percentile(rel, 95)) if rel else None,
        "box_agreement": agree_n / agree_d if agree_d else 1.0,
        "ref_boxes": agree_d,
        "ap50_vs_ref": ap50,
        "swap": swap,
    }
    res["pass"] = bool((res["heat_cosine_min"] or 1) >= 0.99 and heat_abs <= 0.1 and res["box_agreement"] >= 0.95
                       and ap50 >= 0.90 and all(s["ok"] for s in swap))
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
