#!/usr/bin/env python3
"""Inspect the testCYWS pairs: sizes, EXIF orientation, mask frame and change-region statistics.

The tiling defaults come from here: the level-1 tile side is chosen so that the
median ground-truth change component spans about a quarter of a tile, which
matches the object scale CYWS was trained on (COCO objects in a 256 image).

Usage: inspect_dataset.py [ROOT] [--out build/inspect] [--overlays 3]
ROOT defaults to $CYWS_DATA, or data/testCYWS in the repo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

EXIF_ORIENTATION = 0x0112
REPO = Path(__file__).resolve().parent.parent
DEFAULT_DATA = Path(os.environ.get("CYWS_DATA") or REPO / "data" / "testCYWS")


def read(path: Path) -> tuple[np.ndarray, dict]:
    with Image.open(path) as im:
        raw = im.size
        orient = im.getexif().get(EXIF_ORIENTATION, 1)
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(im), {"raw_wh": raw, "wh": im.size, "orientation": orient}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=DEFAULT_DATA, type=Path)
    ap.add_argument("--out", default=REPO / "build" / "inspect", type=Path)
    ap.add_argument("--overlays", type=int, default=3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    stems = sorted(p.name[: -len("_cap.jpg")] for p in (args.root / "images").glob("*_cap.jpg"))
    if not stems:
        print(f"no *_cap.jpg under {args.root}/images", file=sys.stderr)
        return 1
    rows, comp_long_sides, comp_rel = [], [], []
    for i, s in enumerate(stems):
        cap, mc = read(args.root / "images" / f"{s}_cap.jpg")
        ref, mr = read(args.root / "images" / f"{s}_ref.jpg")
        mpath = args.root / "masks" / f"{s}.jpg"
        with Image.open(mpath) as m:
            m_raw = m.size
            m_orient = m.getexif().get(EXIF_ORIENTATION, 1)
            m_modes = m.mode
            mask_gray = np.asarray(ImageOps.exif_transpose(m).convert("L"))
        H, W = cap.shape[:2]
        frame = "canonical" if mask_gray.shape[:2] == (H, W) else (
            "raw" if mask_gray.shape[::-1] == tuple(mc["raw_wh"]) else "other")
        mbin = mask_gray > 127
        if frame != "canonical":
            mbin = cv2.resize(mbin.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        mbin = cv2.morphologyEx(mbin.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mbin.astype(np.uint8), connectivity=8)
        keep = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= 1e-4 * H * W]
        longs = [int(max(stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT])) for k in keep]
        comp_long_sides += longs
        comp_rel += [l / min(H, W) for l in longs]
        # Which image is the mask drawn on? Compare the cap/ref difference inside vs outside the mask
        # (only meaningful if the pair is roughly aligned; reported, not trusted blindly).
        refr = cv2.resize(ref, (W, H), interpolation=cv2.INTER_AREA) if ref.shape[:2] != (H, W) else ref
        d = np.abs(cv2.cvtColor(cap, cv2.COLOR_RGB2GRAY).astype(np.float32)
                   - cv2.cvtColor(refr, cv2.COLOR_RGB2GRAY).astype(np.float32))
        inside = float(d[mbin].mean()) if mbin.any() else float("nan")
        outside = float(d[~mbin].mean())
        uniq = np.unique(mask_gray)
        rows.append({
            "id": s, "cap_wh": mc["wh"], "cap_raw_wh": mc["raw_wh"], "cap_orient": mc["orientation"],
            "ref_wh": mr["wh"], "ref_orient": mr["orientation"], "mask_raw_wh": m_raw, "mask_orient": m_orient,
            "mask_mode": m_modes, "mask_frame": frame, "mask_levels": int(len(uniq)),
            "mask_frac": round(float(mbin.mean()), 5), "components": len(keep), "comp_long_px": longs,
            "diff_in_mask": round(inside, 2), "diff_outside": round(outside, 2),
        })
        if i < args.overlays:
            vis = cap.copy()
            vis[mbin] = (0.5 * vis[mbin] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
            panel = np.concatenate([refr, vis], axis=1)
            scale = 1600 / panel.shape[1]
            panel = cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(args.out / f"{s}_ref_vs_cap_mask.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    for r in rows:
        print(f"{r['id']}  cap {r['cap_wh']} o{r['cap_orient']}  ref {r['ref_wh']} o{r['ref_orient']}  "
              f"mask {r['mask_raw_wh']} {r['mask_mode']} {r['mask_frame']} lv{r['mask_levels']}  "
              f"frac {r['mask_frac']:.4f}  comps {r['components']} {r['comp_long_px']}  "
              f"diff in/out {r['diff_in_mask']}/{r['diff_outside']}")
    summary = {
        "pairs": len(rows),
        "median_component_long_px": float(np.median(comp_long_sides)) if comp_long_sides else None,
        "median_component_rel_min_side": float(np.median(comp_rel)) if comp_rel else None,
        "p10_p90_component_rel": [float(np.percentile(comp_rel, 10)), float(np.percentile(comp_rel, 90))] if comp_rel else None,
        "sizes": sorted({tuple(r["cap_wh"]) for r in rows}),
        "orientations": sorted({r["cap_orient"] for r in rows}),
        "mask_frames": sorted({r["mask_frame"] for r in rows}),
    }
    print(json.dumps(summary, indent=1, default=list))
    (args.out / "inspect.json").write_text(json.dumps({"summary": summary, "pairs": rows}, indent=1, default=list))
    return 0


if __name__ == "__main__":
    sys.exit(main())
