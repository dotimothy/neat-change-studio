"""CenterNet decoding, per-window filtering, cross-window merging, masks,
overlays, metrics and output files for the CYWS app.

Decoding follows mmdet's CenterNet head as used by CYWS: a box is centred on
an integer heat-map peak plus the predicted sub-pixel offset, with the
predicted (w, h) around it, all in 256-pixel model units. Everything after
``filter_window`` works in the canonical frame described in ``common``.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

import tiling
from common import (
    BRANCH_SLICES,
    MODEL_SIZE,
    OUT_CHANNELS,
    AppConfig,
    CywsError,
    Detection,
    Result,
    Window,
)

# Merge constants that the API fixes rather than exposing in AppConfig.
HUGE_SUPPORT_IOU = 0.3      # a huge level>=1 box needs a level-0 box above this IoU
SPLIT_CONTAIN = 0.7         # "level-0 box contains a level>=1 box" threshold
SPLIT_MIN_CHILDREN = 2
SPLIT_MAX_COVER = 0.6       # children covering less than this => coarse box spans objects
WBF_IOU = 0.55
SEAM_CROSS_OVERLAP = 0.5    # seam fragments must overlap this much along the seam

# Metrics constants.
HIT_COVER = 0.5             # one box covers >= 50% of a GT component's pixels
HIT_BBOX_IOU = 0.3          # or box / component-bbox IoU >= 0.3
BOX_ON_GT = 0.3             # a box is "correct" if >= 30% of its area lies on GT

# Mask (diff mode) constants.
DIFF_SIGMA = 2.0
DIFF_MIN_COMPONENT = 0.02   # of the box area
DIFF_MIN_RANGE = 4.0        # blurred |diff| spread below this => Otsu is noise

LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "appeared": (0, 220, 0),
    "disappeared": (230, 40, 40),
    "changed": (255, 200, 0),
}
UNLABELLED_COLOR = (200, 200, 200)
MASK_TINT = (255, 0, 255)


# --------------------------------------------------------------------------
# Box geometry
# --------------------------------------------------------------------------

def _as_boxes(boxes: np.ndarray | Sequence, name: str) -> np.ndarray:
    arr = np.asarray(boxes, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise CywsError(f"{name} must be an (N, 4) box array, got shape {arr.shape}")
    return arr[:, :4]


def _areas(b: np.ndarray) -> np.ndarray:
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def _intersections(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    iw = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
    ih = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    return np.clip(iw, 0, None) * np.clip(ih, 0, None)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in ``a`` (N, 4) against every box in ``b`` (M, 4) -> (N, M).

    Boxes are (x1, y1, x2, y2); degenerate boxes get IoU 0.
    """
    a = _as_boxes(a, "a")
    b = _as_boxes(b, "b")
    inter = _intersections(a, b)
    union = _areas(a)[:, None] + _areas(b)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def containment_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fraction of each ``b`` box that lies inside each ``a`` box -> (N, M).

    ``out[i, j] = area(a[i] ∩ b[j]) / area(b[j])``, so a value near 1 means
    "a[i] contains b[j]". Degenerate ``b`` boxes give 0.
    """
    a = _as_boxes(a, "a")
    b = _as_boxes(b, "b")
    inter = _intersections(a, b)
    area_b = _areas(b)[None, :]
    return np.where(area_b > 0, inter / np.maximum(area_b, 1e-12), 0.0)


def _union_area(boxes: np.ndarray) -> float:
    """Exact area of the union of a few boxes via coordinate compression."""
    if len(boxes) == 0:
        return 0.0
    xs = np.unique(boxes[:, [0, 2]])
    ys = np.unique(boxes[:, [1, 3]])
    covered = np.zeros((len(ys) - 1, len(xs) - 1), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        if x2 <= x1 or y2 <= y1:
            continue
        c0, c1 = np.searchsorted(xs, [x1, x2])
        r0, r1 = np.searchsorted(ys, [y1, y2])
        covered[r0:r1, c0:c1] = True
    cell = np.diff(ys)[:, None] * np.diff(xs)[None, :]
    return float((cell * covered).sum())


def _box_px(box: Iterable[float], h: int, w: int) -> tuple[int, int, int, int]:
    """Integer pixel rect (c0, r0, c1, r1) touched by a float box, clipped to the image."""
    x1, y1, x2, y2 = (float(v) for v in list(box)[:4])
    c0 = min(max(int(math.floor(x1)), 0), w)
    r0 = min(max(int(math.floor(y1)), 0), h)
    c1 = min(max(int(math.ceil(x2)), 0), w)
    r1 = min(max(int(math.ceil(y2)), 0), h)
    return c0, r0, c1, r1


def _det_boxes(dets: Sequence[Detection]) -> np.ndarray:
    if not dets:
        return np.zeros((0, 4), dtype=np.float64)
    return np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], dtype=np.float64)


# --------------------------------------------------------------------------
# Decode and per-window filtering
# --------------------------------------------------------------------------

def decode_centernet(out: np.ndarray, branch: str, thr: float = 0.3, topk: int = 100) -> np.ndarray:
    """Decode one branch of the canonical (10, 256, 256) output into boxes.

    Peaks are pixels equal to their 3x3 max (mmdet's max-pool NMS) with
    heat >= thr. Returns float32 (N, 5) [x1, y1, x2, y2, score] in 256 units,
    sorted by score descending.
    """
    expected = (len(OUT_CHANNELS), MODEL_SIZE, MODEL_SIZE)
    out = np.asarray(out)
    if out.shape != expected:
        raise CywsError(f"decode_centernet expects canonical output {expected}, got {out.shape}")
    if branch not in BRANCH_SLICES:
        raise CywsError(f"unknown branch {branch!r}; expected one of {sorted(BRANCH_SLICES)}")
    if topk < 1:
        raise CywsError(f"topk must be >= 1, got {topk}")
    hc, (wc, hhc), (oxc, oyc) = BRANCH_SLICES[branch]

    heat = np.nan_to_num(out[hc].astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
    hmax = cv2.dilate(heat, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero((heat == hmax) & (heat >= thr))
    if len(ys) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    scores = heat[ys, xs]
    order = np.argsort(-scores, kind="stable")[:topk]
    ys, xs, scores = ys[order], xs[order], scores[order]

    cx = xs + out[oxc, ys, xs].astype(np.float64)
    cy = ys + out[oyc, ys, xs].astype(np.float64)
    w = out[wc, ys, xs].astype(np.float64)
    h = out[hhc, ys, xs].astype(np.float64)
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, scores], axis=1)
    return boxes.astype(np.float32)


def filter_window(dets256: np.ndarray, win: Window, branch: str, cfg: AppConfig) -> list[Detection]:
    """Turn one window's decoded boxes into canonical-frame detections.

    At level >= 1 a box reaching within ``cfg.edge_margin`` of an interior
    (non image-border) window side is a truncated object; an overlapping or
    coarser window sees it whole, so it is dropped. Boxes are then clipped to
    the window and dropped if their short side is below ``cfg.min_box_side``
    (both in 256 units). Clipping before the size test also removes boxes
    that mostly hang outside the window.
    """
    if branch not in BRANCH_SLICES:
        raise CywsError(f"unknown branch {branch!r}")
    d = np.asarray(dets256, dtype=np.float64)
    if d.size == 0:
        return []
    if d.ndim != 2 or d.shape[1] < 5:
        raise CywsError(f"dets256 must be (N, 5), got shape {d.shape}")
    d = d[np.all(np.isfinite(d[:, :5]), axis=1)]

    keep = np.ones(len(d), dtype=bool)
    if win.level >= 1:
        m = cfg.edge_margin
        left, top, right, bottom = win.img_edges
        if not left:
            keep &= d[:, 0] > m
        if not top:
            keep &= d[:, 1] > m
        if not right:
            keep &= d[:, 2] < MODEL_SIZE - m
        if not bottom:
            keep &= d[:, 3] < MODEL_SIZE - m
    d = d[keep]

    d[:, :4] = np.clip(d[:, :4], 0.0, float(MODEL_SIZE))
    short = np.minimum(d[:, 2] - d[:, 0], d[:, 3] - d[:, 1])
    d = d[short >= cfg.min_box_side]
    if len(d) == 0:
        return []

    canon = tiling.window_to_canonical(d[:, :4], win)
    return [
        Detection(
            x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]),
            score=float(s), branch=branch, window_id=win.id, level=win.level,  # type: ignore[arg-type]
        )
        for b, s in zip(canon, d[:, 4])
    ]


def drop_invalid(dets: Sequence[Detection], valid_mask: Optional[np.ndarray],
                 max_invalid_frac: float = 0.5) -> list[Detection]:
    """Drop boxes that lie mostly outside the registered (valid) warp region.

    Those boxes would compare the after image against filled-in after pixels,
    so any score there is meaningless. ``valid_mask=None`` keeps everything.
    """
    if valid_mask is None:
        return list(dets)
    vm = np.asarray(valid_mask)
    if vm.ndim != 2:
        raise CywsError(f"valid_mask must be 2-D, got shape {vm.shape}")
    h, w = vm.shape
    integral = cv2.integral(vm.astype(np.uint8))
    kept: list[Detection] = []
    for det in dets:
        c0, r0, c1, r1 = _box_px(det.box, h, w)
        n = (c1 - c0) * (r1 - r0)
        if n <= 0:
            continue
        valid = integral[r1, c1] - integral[r0, c1] - integral[r1, c0] + integral[r0, c0]
        if 1.0 - valid / n <= max_invalid_frac:
            kept.append(det)
    return kept


# --------------------------------------------------------------------------
# Merge and branch fusion
# --------------------------------------------------------------------------

def _greedy_suppress(dets: list[Detection], nms_iou: float, containment: float) -> list[Detection]:
    """Score-ordered suppression: j goes if IoU > nms_iou or inter/area_j > containment."""
    dets = sorted(dets, key=lambda d: -d.score)
    boxes = _det_boxes(dets)
    iou = iou_matrix(boxes, boxes)
    cont = containment_matrix(boxes, boxes)
    alive = np.ones(len(dets), dtype=bool)
    for i in range(len(dets)):
        if not alive[i]:
            continue
        later = np.arange(i + 1, len(dets))
        hit = (iou[i, later] > nms_iou) | (cont[i, later] > containment)
        alive[later[hit]] = False
    return [d for d, a in zip(dets, alive) if a]


def _wbf(dets: list[Detection], iou_thr: float = WBF_IOU) -> list[Detection]:
    """Weighted box fusion: score-weighted average box per IoU cluster, max score."""
    dets = sorted(dets, key=lambda d: -d.score)
    members: list[list[Detection]] = []
    fused: list[np.ndarray] = []
    for det in dets:
        if fused:
            ious = iou_matrix(det.box[None, :], np.array(fused))[0]
            best = int(np.argmax(ious))
            if ious[best] > iou_thr:
                members[best].append(det)
                ws = np.array([m.score for m in members[best]])
                bs = _det_boxes(members[best])
                fused[best] = (bs * ws[:, None]).sum(0) / max(ws.sum(), 1e-12)
                continue
        members.append([det])
        fused.append(det.box)
    return [
        replace(group[0], x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]))
        for group, b in zip(members, fused)
    ]


def _seam_cuts(det: Detection, win: Window, margin256: float) -> tuple[bool, bool, bool, bool]:
    """Which interior sides (left, top, right, bottom) of its window a box reaches."""
    mx, my = margin256 * win.w / MODEL_SIZE, margin256 * win.h / MODEL_SIZE
    left, top, right, bottom = win.img_edges
    return (not left and det.x1 <= win.x0 + mx, not top and det.y1 <= win.y0 + my,
            not right and det.x2 >= win.x1 - mx, not bottom and det.y2 >= win.y1 - my)


def _join_seam_fragments(l0: list[Detection], windows_by_id: dict[int, Window],
                         cfg: AppConfig) -> list[Detection]:
    """Repair level-0 boxes cut by the seams between elongated-image squares.

    An image with aspect > 1.5 gets several level-0 squares, and no coarser
    window sees an object that crosses their shared side whole, so it decodes
    as one partial box per square. Neither NMS nor containment merges those
    (their overlap is only the strip the squares share), so here a cut box is
    dropped when an uncut box from another level-0 square contains it, and
    otherwise chained with the cut boxes of neighbouring squares that face it
    across the seam and overlap it along the seam; each chain becomes one box
    (union extent, best score, provenance of the best member).
    """
    wins = {d.window_id: windows_by_id.get(d.window_id) for d in l0}
    if len({wid for wid, w in wins.items() if w is not None}) < 2:
        return l0
    cuts = [_seam_cuts(d, wins[d.window_id], cfg.edge_margin) if wins[d.window_id] else (False,) * 4
            for d in l0]
    cut = np.array([any(c) for c in cuts], dtype=bool)
    if not cut.any():
        return l0
    boxes = _det_boxes(l0)
    cont = containment_matrix(boxes, boxes)          # cont[i, j]: share of j inside i
    wid = np.array([d.window_id for d in l0])
    covered = np.array([
        cut[j] and bool(np.any(~cut & (wid != wid[j]) & (cont[:, j] > cfg.containment)))
        for j in range(len(l0))
    ])

    parent = list(range(len(l0)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    live = [i for i in range(len(l0)) if cut[i] and not covered[i]]
    for ai, i in enumerate(live):
        for j in live[ai + 1:]:
            if wid[i] == wid[j]:
                continue
            wi, wj = wins[l0[i].window_id], wins[l0[j].window_id]
            if wi.x0 != wj.x0:                         # side-by-side squares: seam is vertical
                a, b = (i, j) if wi.x0 < wj.x0 else (j, i)
                facing = cuts[a][2] and cuts[b][0]
                lo, hi = 1, 3
            else:                                      # stacked squares: seam is horizontal
                a, b = (i, j) if wi.y0 < wj.y0 else (j, i)
                facing = cuts[a][3] and cuts[b][1]
                lo, hi = 0, 2
            along = 1 - lo                             # the axis crossing the seam
            # a's far side reaches b's near side (cut boxes may stop up to a margin short).
            slack = cfg.edge_margin * (wi.w + wj.w) / MODEL_SIZE
            touch = boxes[a, along + 2] + slack >= boxes[b, along]
            cross = min(boxes[i, hi], boxes[j, hi]) - max(boxes[i, lo], boxes[j, lo])
            extent = min(boxes[i, hi] - boxes[i, lo], boxes[j, hi] - boxes[j, lo])
            if facing and touch and extent > 0 and cross / extent > SEAM_CROSS_OVERLAP:
                parent[root(i)] = root(j)

    groups: dict[int, list[int]] = {}
    for i in live:
        groups.setdefault(root(i), []).append(i)
    out = [d for i, d in enumerate(l0) if not cut[i]]
    for members in groups.values():
        best = max(members, key=lambda k: l0[k].score)
        m = boxes[members]
        out.append(replace(l0[best], x1=float(m[:, 0].min()), y1=float(m[:, 1].min()),
                           x2=float(m[:, 2].max()), y2=float(m[:, 3].max())))
    return out


def _merge_branch(dets: list[Detection], windows_by_id: dict[int, Window],
                  has_level0: bool, cfg: AppConfig) -> list[Detection]:
    if not dets:
        return []
    # (0) Seam rule: rejoin objects cut by the seams between level-0 squares.
    l0 = _join_seam_fragments([d for d in dets if d.level == 0], windows_by_id, cfg)
    fine = [d for d in dets if d.level >= 1]

    # (1) Huge-box rule: a fine box filling most of its window is usually a
    # tile-level hallucination unless the full view agrees.
    if has_level0 and fine:
        l0_boxes = _det_boxes(l0)
        kept_fine = []
        for d in fine:
            win = windows_by_id.get(d.window_id)
            if win is None:
                kept_fine.append(d)
                continue
            wbox = np.array([[win.x0, win.y0, win.x1, win.y1]], dtype=np.float64)
            cover = _intersections(d.box[None, :], wbox)[0, 0] / max(win.w * win.h, 1)
            if cover <= cfg.huge_box_frac:
                kept_fine.append(d)
            elif len(l0_boxes) and iou_matrix(d.box[None, :], l0_boxes).max() > HUGE_SUPPORT_IOU:
                kept_fine.append(d)
        fine = kept_fine

    # (2) Split rule: a coarse box that spans several separate fine detections
    # with lots of empty space between them is replaced by those detections.
    if l0 and len(fine) >= SPLIT_MIN_CHILDREN:
        fine_boxes = _det_boxes(fine)
        cont = containment_matrix(_det_boxes(l0), fine_boxes)
        kept_l0 = []
        for i, d in enumerate(l0):
            children = fine_boxes[cont[i] > SPLIT_CONTAIN]
            if len(children) >= SPLIT_MIN_CHILDREN and d.area > 0:
                clipped = children.copy()
                clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], d.x1, d.x2)
                clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], d.y1, d.y2)
                if _union_area(clipped) / d.area < SPLIT_MAX_COVER:
                    continue
            kept_l0.append(d)
        l0 = kept_l0

    # (3) Cross-window duplicates and nested boxes.
    pool = l0 + fine
    if cfg.merge == "wbf":
        pool = _wbf(pool)
    elif cfg.merge != "nms":
        raise CywsError(f"unknown merge mode {cfg.merge!r}; expected 'nms' or 'wbf'")
    merged = _greedy_suppress(pool, cfg.nms_iou, cfg.containment)

    # (4) Cap.
    return merged[: max(int(cfg.max_boxes), 0)]


def merge(dets: Sequence[Detection], windows: Sequence[Window], cfg: AppConfig) -> list[Detection]:
    """Merge detections from all windows, one branch at a time.

    Rules in order: seam rule (level-0 boxes cut by the seam between the
    squares of an elongated image are rejoined), huge-box rule, split rule,
    greedy suppression (or WBF
    followed by the same suppression, which then only removes nested boxes
    and the 0.5-0.55 IoU band), cap at ``cfg.max_boxes`` per branch. The huge
    -box rule is skipped when no level-0 window was planned, since there is
    nothing to support it. Returns detections sorted by score descending.
    """
    unknown = {d.branch for d in dets} - set(BRANCH_SLICES)
    if unknown:
        raise CywsError(f"detections with unknown branch {sorted(unknown)}")
    windows_by_id = {w.id: w for w in windows}
    has_level0 = any(w.level == 0 for w in windows)
    out: list[Detection] = []
    for branch in BRANCH_SLICES:
        out += _merge_branch([d for d in dets if d.branch == branch], windows_by_id, has_level0, cfg)
    return sorted(out, key=lambda d: -d.score)


def fuse_branches(after_dets: Sequence[Detection], before_dets: Sequence[Detection],
                  iou: float = 0.5) -> list[Detection]:
    """Label changes by pairing the two branches.

    Pairs are matched greedily by IoU (highest first, each box used once).
    A matched pair (IoU > iou) becomes one "changed" detection with the after
    box and the max score; unmatched after boxes are "appeared", unmatched
    before boxes "disappeared". Returns new objects sorted by score.
    """
    a_boxes, b_boxes = _det_boxes(after_dets), _det_boxes(before_dets)
    ious = iou_matrix(a_boxes, b_boxes)
    used_a: set[int] = set()
    used_b: set[int] = set()
    out: list[Detection] = []
    if ious.size:
        for flat in np.argsort(-ious, axis=None, kind="stable"):
            i, j = (int(v) for v in np.unravel_index(flat, ious.shape))
            if ious[i, j] <= iou:
                break
            if i in used_a or j in used_b:
                continue
            used_a.add(i)
            used_b.add(j)
            a, b = after_dets[i], before_dets[j]
            out.append(replace(a, score=max(a.score, b.score), label="changed"))
    out += [replace(d, label="appeared") for k, d in enumerate(after_dets) if k not in used_a]
    out += [replace(d, label="disappeared") for k, d in enumerate(before_dets) if k not in used_b]
    return sorted(out, key=lambda d: -d.score)


# --------------------------------------------------------------------------
# Masks
# --------------------------------------------------------------------------

def _check_hw(hw: Sequence[int]) -> tuple[int, int]:
    if len(hw) < 2 or int(hw[0]) <= 0 or int(hw[1]) <= 0:
        raise CywsError(f"hw must be (H, W) with positive sizes, got {tuple(hw)}")
    return int(hw[0]), int(hw[1])


def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32)
    return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2GRAY).astype(np.float32)


def _diff_box_mask(diff: np.ndarray, rect: tuple[int, int, int, int], pad: int) -> Optional[np.ndarray]:
    """Otsu-segment |diff| inside one box rect; None when nothing credible survives."""
    c0, r0, c1, r1 = rect
    h, w = diff.shape
    pc0, pr0 = max(c0 - pad, 0), max(r0 - pad, 0)
    pc1, pr1 = min(c1 + pad, w), min(r1 + pad, h)
    blurred = cv2.GaussianBlur(diff[pr0:pr1, pc0:pc1], (0, 0), DIFF_SIGMA)
    patch = blurred[r0 - pr0:r1 - pr0, c0 - pc0:c1 - pc0]
    if patch.size == 0 or float(patch.max() - patch.min()) < DIFF_MIN_RANGE:
        return None
    u8 = np.clip(patch, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = DIFF_MIN_COMPONENT * patch.size
    good = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] > min_area]
    if not good:
        return None
    return np.isin(labels, good)


def make_mask(dets: Sequence[Detection], hw: Sequence[int], mode: str = "box",
              after: Optional[np.ndarray] = None, before_aligned: Optional[np.ndarray] = None,
              valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Rasterise detections into a bool (H, W) change mask.

    CYWS has no segmentation head, so "box" fills the boxes. "diff" segments
    the blurred |gray diff| inside each box with Otsu + morphology and keeps
    components > 2% of the box, falling back to the filled box when nothing
    survives. "diff" needs both images; without them it behaves like "box".
    """
    h, w = _check_hw(hw)
    if mode not in ("box", "diff"):
        raise CywsError(f"unknown mask mode {mode!r}; expected 'box' or 'diff'")
    mask = np.zeros((h, w), dtype=bool)
    use_diff = mode == "diff" and after is not None and before_aligned is not None
    diff = None
    if use_diff:
        if after.shape[:2] != (h, w) or before_aligned.shape[:2] != (h, w):
            raise CywsError(
                f"diff mask needs images of shape {(h, w)}, got after {after.shape[:2]} "
                f"and before_aligned {before_aligned.shape[:2]}"
            )
        diff = np.abs(_gray(after) - _gray(before_aligned))
        if valid_mask is not None:
            if valid_mask.shape != (h, w):
                raise CywsError(f"valid_mask shape {valid_mask.shape} != {(h, w)}")
            diff[~valid_mask.astype(bool)] = 0.0
    pad = int(math.ceil(3 * DIFF_SIGMA))
    for det in dets:
        c0, r0, c1, r1 = _box_px(det.box, h, w)
        if c1 <= c0 or r1 <= r0:
            continue
        seg = _diff_box_mask(diff, (c0, r0, c1, r1), pad) if diff is not None else None
        if seg is None:
            mask[r0:r1, c0:c1] = True
        else:
            mask[r0:r1, c0:c1] |= seg
    return mask


# --------------------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------------------

def _draw_label(img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int],
                scale: float, thick: int) -> None:
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    y = max(y, th + base + 2)
    x = min(max(x, 0), max(img.shape[1] - tw - 4, 0))
    cv2.rectangle(img, (x, y - th - base - 2), (x + tw + 4, y + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 2, y - base), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def overlay(after: np.ndarray, result: Result, heat_alpha: float = 0.35) -> np.ndarray:
    """Visualise a result on the after image (uint8 RGB, same shape as ``after``).

    The heat canvas is blended in proportion to its value, so cold areas keep
    their original colours; the change mask is tinted; boxes are coloured by
    label with "label score" text; a legend sits in the top-left corner.
    """
    img = np.asarray(after)
    if img.ndim != 3 or img.shape[2] != 3:
        raise CywsError(f"after must be (H, W, 3) RGB, got shape {img.shape}")
    h, w = img.shape[:2]
    # uint8 OpenCV blends (multi-threaded): float32 numpy over a 3 MP frame took
    # ~250 ms on the board's A65.
    out = np.ascontiguousarray(img, dtype=np.uint8).copy()

    heat = result.heat
    if heat is not None and np.size(heat) and float(np.max(heat)) > 0 and heat_alpha > 0:
        heat_full = cv2.resize(np.clip(heat, 0, 1).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        colour = cv2.applyColorMap((heat_full * 255).astype(np.uint8), cv2.COLORMAP_JET)
        colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)
        a = cv2.multiply(heat_full, float(heat_alpha))
        out = cv2.blendLinear(out, colour, cv2.subtract(1.0, a), a)

    mask = result.mask
    if mask is not None and np.size(mask):
        if mask.shape != (h, w):
            raise CywsError(f"result.mask shape {mask.shape} != after shape {(h, w)}")
        m = np.ascontiguousarray(mask, dtype=bool).view(np.uint8)
        if m.any():
            # 0.75 * pixel + 0.25 * tint with per-channel scalars (a numpy broadcast fill of a
            # full tint image alone took ~120 ms on the A65).
            tinted = cv2.add(cv2.multiply(out, (0.75, 0.75, 0.75, 0.0)),
                             tuple(0.25 * float(c) for c in MASK_TINT) + (0.0,))
            cv2.copyTo(tinted, m, out)
    longest = max(h, w)
    thick = max(1, round(longest / 600))
    scale = max(0.4, longest / 1800)
    txt_thick = max(1, thick // 2)
    for det in result.detections:
        colour = LABEL_COLORS.get(det.label or "", UNLABELLED_COLOR)
        p1 = (int(round(det.x1)), int(round(det.y1)))
        p2 = (int(round(det.x2)) - 1, int(round(det.y2)) - 1)
        cv2.rectangle(out, p1, p2, colour, thick, cv2.LINE_AA)
        _draw_label(out, f"{det.label or det.branch} {det.score:.2f}", (p1[0], p1[1] - 2), colour, scale, txt_thick)

    (_, th), base = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, scale, txt_thick)
    row = th + base + 6
    for k, (name, colour) in enumerate(LABEL_COLORS.items()):
        y = 6 + k * row
        cv2.rectangle(out, (6, y), (6 + row - 4, y + row - 4), colour, -1)
        _draw_label(out, name, (6 + row, y + row - 4), colour, scale, txt_thick)
    return out


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def _ratio(num: float, den: float, empty: float) -> float:
    return float(num) / float(den) if den > 0 else float(empty)


def metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, dets: Sequence[Detection]) -> dict:
    """Pixel and component metrics of a prediction against a GT mask.

    Empty-denominator conventions keep every value a float: with no GT and
    no prediction everything is 1.0; precision (pixel or box) with no
    prediction is 1.0 only if GT is empty too, else 0.0; recall and
    component_recall with no GT are 1.0.
    """
    pred = np.asarray(pred_mask).astype(bool)
    gt = np.asarray(gt_mask).astype(bool)
    if pred.shape != gt.shape or gt.ndim != 2:
        raise CywsError(f"pred_mask {pred.shape} and gt_mask {gt.shape} must be the same 2-D shape")
    h, w = gt.shape
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    gt_empty = (tp + fn) == 0

    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8), connectivity=8)
    n_comp = n_lab - 1
    comp_area = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    comp_bbox = np.stack([
        stats[1:, cv2.CC_STAT_LEFT], stats[1:, cv2.CC_STAT_TOP],
        stats[1:, cv2.CC_STAT_LEFT] + stats[1:, cv2.CC_STAT_WIDTH],
        stats[1:, cv2.CC_STAT_TOP] + stats[1:, cv2.CC_STAT_HEIGHT],
    ], axis=1).astype(np.float64) if n_comp else np.zeros((0, 4))

    hit = np.zeros(n_comp, dtype=bool)
    boxes_on_gt = 0
    boxes = _det_boxes(dets)
    bbox_iou = iou_matrix(boxes, comp_bbox) if n_comp else np.zeros((len(boxes), 0))
    for k, det in enumerate(dets):
        c0, r0, c1, r1 = _box_px(det.box, h, w)
        n_px = (c1 - c0) * (r1 - r0)
        if n_px <= 0:
            continue
        sub = labels[r0:r1, c0:c1]
        if n_comp:
            counts = np.bincount(sub.ravel(), minlength=n_lab)[1:]
            hit |= counts / comp_area >= HIT_COVER
            hit |= bbox_iou[k] >= HIT_BBOX_IOU
        if np.count_nonzero(sub) / n_px >= BOX_ON_GT:
            boxes_on_gt += 1

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou": _ratio(tp, tp + fp + fn, 1.0),
        "precision": _ratio(tp, tp + fp, 1.0 if gt_empty else 0.0),
        "recall": _ratio(tp, tp + fn, 1.0),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn, 1.0),
        "gt_components": int(n_comp),
        "gt_components_hit": int(hit.sum()),
        "component_recall": _ratio(int(hit.sum()), n_comp, 1.0),
        "num_boxes": len(dets),
        "box_precision": _ratio(boxes_on_gt, len(dets), 1.0 if gt_empty else 0.0),
    }


# --------------------------------------------------------------------------
# Output files
# --------------------------------------------------------------------------

def _jsonable(obj):
    """Best-effort conversion of meta values (numpy, Path, tuples, NaN) to strict JSON."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _window_dict(win: Window) -> dict:
    return {"id": win.id, "level": win.level, "x0": win.x0, "y0": win.y0,
            "w": win.w, "h": win.h, "img_edges": list(win.img_edges)}


def _registration_dict(result: Result) -> Optional[dict]:
    reg = result.registration
    if reg is None:
        return None
    vm = np.asarray(reg.valid_mask)
    return {
        "ok": bool(reg.ok),
        "method": reg.method,
        "reason": reg.reason,
        "inliers": int(reg.inliers),
        "inlier_ratio": round(float(reg.inlier_ratio), 4),
        "H": [[float(v) for v in row] for row in np.asarray(reg.H, dtype=np.float64)],
        "valid_frac": round(float(vm.mean()), 4) if vm.size else None,
    }


def _scale_of(result: Result) -> tuple[float, float]:
    """(sx, sy) original after pixels per canonical pixel from result.meta; (1, 1) if absent."""
    meta = result.meta if isinstance(result.meta, dict) else {}
    sc = meta.get("scale")
    try:
        sx, sy = (float(v) for v in sc)
    except (TypeError, ValueError):
        return 1.0, 1.0
    if not (np.isfinite(sx) and np.isfinite(sy) and sx > 0 and sy > 0):
        return 1.0, 1.0
    return sx, sy


def save_outputs(result: Result, after: np.ndarray, out_dir: str | Path, *,
                 save_heat: bool = False) -> dict[str, str]:
    """Write ``<id>_overlay.jpg``, ``<id>_mask.png``, ``<id>.json`` and, with
    ``save_heat``, ``<id>_heat.png``. Returns {kind: path}.

    Images are written at the canonical (working) resolution. Each JSON
    detection has "box" (canonical frame) and "box_orig" (original after
    pixels, box * result.meta["scale"]).
    """
    out = Path(out_dir)
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CywsError(f"cannot create output directory {out}: {exc}") from exc
    stem = re.sub(r"[^\w.\-]+", "_", result.pair_id) or "pair"
    paths: dict[str, str] = {}

    def _write_img(kind: str, name: str, img: np.ndarray, params: Sequence[int] = ()) -> None:
        path = out / name
        if not cv2.imwrite(str(path), img, list(params)):
            raise CywsError(f"failed to write {path}")
        paths[kind] = str(path)

    vis = overlay(after, result)
    _write_img("overlay", f"{stem}_overlay.jpg", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
               (cv2.IMWRITE_JPEG_QUALITY, 92))
    _write_img("mask", f"{stem}_mask.png", result.mask.astype(np.uint8) * 255)
    if save_heat and result.heat is not None and np.size(result.heat):
        _write_img("heat", f"{stem}_heat.png", (np.clip(result.heat, 0, 1) * 255).round().astype(np.uint8))

    h, w = np.asarray(after).shape[:2]
    # Boxes stay in the canonical (working-resolution) frame; "box_orig" maps them
    # to the original after image with the scale load_pair recorded.
    scale = _scale_of(result)
    meta_orig = result.meta.get("orig_size") if isinstance(result.meta, dict) else None
    ow, oh = (int(v) for v in meta_orig) if meta_orig is not None else (round(w * scale[0]), round(h * scale[1]))
    doc = {
        "pair_id": result.pair_id,
        "image_size": {"w": int(w), "h": int(h)},
        "orig_image_size": {"w": int(ow), "h": int(oh)},
        "scale": [float(scale[0]), float(scale[1])],
        "num_detections": len(result.detections),
        "detections": [d.to_dict(scale) for d in result.detections],
        "windows": [_window_dict(win) for win in result.windows],
        "registration": _registration_dict(result),
        "timings_ms": {k: round(float(v), 3) for k, v in result.timings_ms.items()},
        "meta": result.meta,
        "files": dict(paths),
    }
    json_path = out / f"{stem}.json"
    try:
        json_path.write_text(json.dumps(_jsonable(doc), indent=2, allow_nan=False))
    except OSError as exc:
        raise CywsError(f"failed to write {json_path}: {exc}") from exc
    paths["json"] = str(json_path)
    return paths
