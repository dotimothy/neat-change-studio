"""C-3PO change segmentation in Change Studio's Result format.

C-3PO (github.com/DoctorKey/C-3PO, MIT) takes a before and an after image, 256 x 256 each,
and returns change logits at 64 x 64. The whole photo (and, with levels 1/2, overlapping
square tiles of 0.6 / 0.4 of the short side, like app/zeroshot.py) is squashed to 256 and
run once per window; each window's change probability, 1 - P(no change), is upsampled to
the window and the windows are combined by a per-pixel max. With gated tiles a tile only
runs where the whole-photo map reaches `gate`, and only counts there.

The mask is probability >= thr; its connected regions of at least min_area become boxes,
scored by their peak probability. The released checkpoints are binary (change / no change),
so every box is labelled "changed". Everything is in the canonical (after photo) frame from
common.py, decoded by the same ingestion.load_pair as CYWS, so ground truth, metrics and the
page's layers work unchanged.

Defaults are the tuned setting on testCYWS (FP32 ONNX, both checkpoints): thr 0.20, 0.6 tiles
gated at 0.05, min_area 0.002. The same threshold suits PCD and ChangeSim (ChangeSim's F1
moves by under 0.005 from 0.10 to 0.50), so there is one set of presets for both.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

import ingestion
import registration
from c3po_backends import change_logit, sigmoid
from common import CywsError, Detection, Pair, PairItem, RegResult, Result, Window
from dinov2_change import resize_square, tile_windows

TIMING_KEYS = ("load", "register", "stream", "postprocess", "total")
TILE_FRAC = {1: 0.6, 2: 0.4}            # tile side per level, as a share of the short side
HEAT_AT_THR = 0.3                        # display heat value at the threshold


@dataclass
class C3POConfig:
    levels: tuple[int, ...] = (0, 1)     # 0 whole photo, 1 tiles of 0.6, 2 tiles of 0.4
    tiles: str = "gated"                 # gated: tiles count only where the whole-photo map >= gate
    thr: float = 0.20
    register: bool = False
    mask_mode: str = "box"               # accepted for the shared controls; the mask is per pixel anyway
    verify_change: Optional[bool] = None  # accepted and ignored
    gate: float = 0.05
    overlap: float = 0.25
    min_area: float = 0.002
    work_long: int = 1024
    allow_squash: bool = False
    reg_method: str = "sift"


def validate_config(cfg: C3POConfig) -> None:
    if not cfg.levels or 0 not in cfg.levels or any(lv not in (0, 1, 2) for lv in cfg.levels) \
            or len(set(cfg.levels)) != len(cfg.levels):
        raise CywsError(f"levels must be distinct values from 0, 1, 2 and include 0; got {cfg.levels}")
    if cfg.tiles not in ("full", "gated"):
        raise CywsError(f"tiles must be 'full' or 'gated'; got {cfg.tiles!r}")
    if not 0.0 < cfg.thr < 1.0:
        raise CywsError(f"thr must be in (0, 1); got {cfg.thr}")
    if not isinstance(cfg.work_long, int) or (cfg.work_long != 0 and cfg.work_long < 256):
        raise CywsError(f"work_long must be 0 or an integer >= 256; got {cfg.work_long!r}")


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def regions(prob: np.ndarray, thr: float, min_area: float) -> tuple[np.ndarray, list[tuple]]:
    """Mask of prob >= thr without regions under min_area (a share of the frame), and the
    kept regions as (x1, y1, x2, y2, peak, label image index)."""
    h, w = prob.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats((prob >= thr).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area / float(h * w) < min_area:
            continue
        keep[i] = True
        sl = np.s_[y:y + bh, x:x + bw]
        peak = float(prob[sl][labels[sl] == i].max())
        out.append((float(x), float(y), float(x + bw), float(y + bh), peak, i))
    out.sort(key=lambda r: -r[4])
    return keep[labels], out


class C3POApp:
    """One pair at a time on a C-3PO backend (owned by the caller)."""

    def __init__(self, backend: Any, cfg: Optional[C3POConfig] = None) -> None:
        self.backend = backend
        self.cfg = cfg if cfg is not None else C3POConfig()
        validate_config(self.cfg)

    def load(self, item: PairItem) -> Pair:
        t0 = time.perf_counter()
        pair = ingestion.load_pair(item, allow_squash=self.cfg.allow_squash, register=self.cfg.register,
                                   work_long=self.cfg.work_long)
        pair.meta["load_ms"] = _ms(t0)
        return pair

    def _prob(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """Change probability over one window, at the window's own size."""
        h, w = after.shape[:2]
        s = self.backend.size
        lg = self.backend.infer(resize_square(before, s), resize_square(after, s))
        return self._upsample(lg, w, h)

    @staticmethod
    def _upsample(lg: np.ndarray, w: int, h: int) -> np.ndarray:
        return sigmoid(cv2.resize(change_logit(lg), (w, h), interpolation=cv2.INTER_LINEAR))

    def _probs(self, crops: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
        """_prob for several windows in one batch, so the backend can keep pairs in flight."""
        s = self.backend.size
        batch = [(resize_square(b, s), resize_square(a, s)) for b, a in crops]
        many = getattr(self.backend, "infer_many", None)
        lgs = many(batch) if many is not None else [self.backend.infer(b, a) for b, a in batch]
        return [self._upsample(lg, a.shape[1], a.shape[0]) for lg, (_, a) in zip(lgs, crops)]

    def process(self, pair: Pair) -> Result:
        cfg = self.cfg
        t_start = time.perf_counter()
        timings = {k: 0.0 for k in TIMING_KEYS}
        timings["load"] = float(pair.meta.get("load_ms", 0.0))
        after = pair.after
        if after.ndim != 3 or after.shape[2] != 3 or after.dtype != np.uint8:
            raise CywsError(f"{pair.pair_id}: after must be uint8 (H, W, 3); got {after.dtype} {after.shape}")
        H, W = after.shape[:2]

        t0 = time.perf_counter()
        reg: Optional[RegResult] = None
        valid = None
        if cfg.register:
            reg = registration.register(pair.before, after, method=cfg.reg_method)
            before = registration.warp_before(pair.before, reg, after)
            valid = np.asarray(reg.valid_mask, bool)
        else:
            before = pair.before if pair.before.shape == after.shape else ingestion.resize_to(pair.before, W, H)
        before = np.ascontiguousarray(before)
        timings["register"] = _ms(t0)

        # whole photo first; with gated tiles its map decides which tiles are worth a run
        t0 = time.perf_counter()
        base = self._prob(before, after)
        windows = [Window(id=0, level=0, x0=0, y0=0, w=W, h=H, img_edges=(True, True, True, True))]
        maps = [base]
        tile = np.full((H, W), -1.0, np.float32)
        gated_out, by_level = 0, {0: 1}
        gated = cfg.tiles == "gated"
        todo = []                                    # (level, window, slice) for the tiles worth a run
        for level in sorted(lv for lv in cfg.levels if lv > 0):
            for win in tile_windows(H, W, frac=TILE_FRAC[level], overlap=cfg.overlap):
                y1, x1 = min(H, win.y0 + win.side), min(W, win.x0 + win.side)
                sl = np.s_[win.y0:y1, win.x0:x1]
                if gated and float(base[sl].max()) < cfg.gate:
                    gated_out += 1                   # would add nothing: the gate keeps the whole-photo map there
                    continue
                todo.append((level, win, x1, y1, sl))
        probs = self._probs([(before[sl], after[sl]) for *_, sl in todo]) if todo else []
        for (level, win, x1, y1, sl), p in zip(todo, probs):
            np.maximum(tile[sl], p, out=tile[sl])
            windows.append(Window(id=len(windows), level=level, x0=win.x0, y0=win.y0, w=x1 - win.x0,
                                  h=y1 - win.y0, img_edges=(win.x0 == 0, win.y0 == 0, x1 >= W, y1 >= H)))
            maps.append(p)
            by_level[level] = by_level.get(level, 0) + 1
        timings["stream"] = _ms(t0)

        t0 = time.perf_counter()
        if gated:
            tile = np.where(base >= cfg.gate, tile, -1.0)
        prob = np.maximum(base, tile)
        if valid is not None:
            prob = np.where(valid, prob, 0.0).astype(np.float32)
        mask, found = regions(prob, cfg.thr, cfg.min_area)
        dets = []
        for x1, y1, x2, y2, peak, _ in found:
            wid, level = self._strongest((x1, y1, x2, y2), windows, maps)
            dets.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=peak, branch="after",
                                  window_id=wid, level=level, label="changed"))
        heat = prob * np.float32(HEAT_AT_THR / cfg.thr)
        heat = cv2.resize(heat.astype(np.float32), (max(1, W // 4), max(1, H // 4)), interpolation=cv2.INTER_AREA)
        counts = {"appeared": 0, "disappeared": 0, "changed": len(dets)}
        timings["postprocess"] = _ms(t0)
        timings["total"] = timings["load"] + _ms(t_start)
        meta = {
            "counts": counts, "windows_run": len(windows), "windows_by_level": {str(k): v for k, v in by_level.items()},
            "gated_out": gated_out, "pair": pair.pair_id, "backend": getattr(self.backend, "name", "?"),
            "work_size": [W, H], "scale": [float(v) for v in pair.meta.get("scale", (1.0, 1.0))],
        }
        return Result(pair_id=pair.pair_id, detections=dets, mask=mask, heat=heat, windows=windows,
                      registration=reg, timings_ms={k: round(v, 3) for k, v in timings.items()},
                      before_aligned=before if cfg.register else None, meta=meta)

    @staticmethod
    def _strongest(box, windows: list[Window], maps: list[np.ndarray]) -> tuple[int, int]:
        """(window id, level) of the window whose probability peaks highest inside the box."""
        best, pick = -np.inf, (0, 0)
        bx1, by1, bx2, by2 = (int(v) for v in box)
        for win, p in zip(windows, maps):
            x1, y1 = max(bx1, win.x0), max(by1, win.y0)
            x2, y2 = min(bx2, win.x0 + win.w), min(by2, win.y0 + win.h)
            if x2 <= x1 or y2 <= y1:
                continue
            v = float(p[y1 - win.y0:y2 - win.y0, x1 - win.x0:x2 - win.x0].max())
            if v > best:
                best, pick = v, (win.id, win.level)
        return pick
