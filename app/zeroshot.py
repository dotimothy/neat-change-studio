"""Zero-shot change detection with DINOv2 features, in Change Studio's Result format.

The whole photo (and, with levels 1/2, overlapping square tiles of 0.6 / 0.4 of the short
side) goes through the feature backend for both photos; app/dinov2_change.py turns the
patch maps into a per-pixel change mask and labelled boxes. Everything is in the
canonical (after photo) frame from common.py, decoded by the same ingestion.load_pair
as CYWS, so ground truth, metrics and the page's layers work unchanged.

Defaults are the tuned setting on testCYWS (DINOv2/examples/change_detect.py): radius 0,
border 1, blur 1, min_area 0.002, thr 0.40, 0.6 tiles gated at 0.2.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

import dinov2_change as change
import ingestion
import registration
from common import CywsError, Detection, Pair, PairItem, RegResult, Result, Window
from feature_backends import split

TIMING_KEYS = ("load", "register", "stream", "postprocess", "total")
TILE_FRAC = {1: 0.6, 2: 0.4}            # tile side per level, as a share of the short side
HEAT_AT_THR = 0.3                        # display heat value at the threshold


@dataclass
class ZeroShotConfig:
    levels: tuple[int, ...] = (0, 1)     # 0 whole photo, 1 tiles of 0.6, 2 tiles of 0.4
    tiles: str = "gated"                 # gated: tiles count only where the whole-photo map >= gate
    thr: float = 0.40
    register: bool = False
    mask_mode: str = "box"               # accepted for the shared controls; the mask is per pixel anyway
    verify_change: Optional[bool] = None  # accepted and ignored
    gate: float = 0.2
    overlap: float = 0.25
    radius: int = 0
    border: int = 1
    blur: float = 1.0
    min_area: float = 0.002
    label_radius: int = 2
    work_long: int = 1024
    allow_squash: bool = False
    reg_method: str = "sift"


def validate_config(cfg: ZeroShotConfig) -> None:
    if not cfg.levels or 0 not in cfg.levels or any(lv not in (0, 1, 2) for lv in cfg.levels) \
            or len(set(cfg.levels)) != len(cfg.levels):
        raise CywsError(f"levels must be distinct values from 0, 1, 2 and include 0; got {cfg.levels}")
    if cfg.tiles not in ("full", "gated"):
        raise CywsError(f"tiles must be 'full' or 'gated'; got {cfg.tiles!r}")
    if not 0.0 < cfg.thr <= 1.0:
        raise CywsError(f"thr must be in (0, 1]; got {cfg.thr}")
    if not isinstance(cfg.work_long, int) or (cfg.work_long != 0 and cfg.work_long < 256):
        raise CywsError(f"work_long must be 0 or an integer >= 256; got {cfg.work_long!r}")


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


class ZeroShotApp:
    """One pair at a time on a DINOv2 feature backend (owned by the caller)."""

    def __init__(self, backend: Any, cfg: Optional[ZeroShotConfig] = None) -> None:
        self.backend = backend
        self.cfg = cfg if cfg is not None else ZeroShotConfig()
        validate_config(self.cfg)

    def load(self, item: PairItem) -> Pair:
        t0 = time.perf_counter()
        pair = ingestion.load_pair(item, allow_squash=self.cfg.allow_squash, register=self.cfg.register,
                                   work_long=self.cfg.work_long)
        pair.meta["load_ms"] = _ms(t0)
        return pair

    def _change_cfg(self) -> change.ChangeConfig:
        c = self.cfg
        return change.ChangeConfig(radius=c.radius, thr=c.thr, min_area=c.min_area, blur=c.blur,
                                   border=c.border, label_radius=c.label_radius)

    def _feats(self, img: np.ndarray) -> np.ndarray:
        return split(self.backend.features(change.resize_square(img, self.backend.size)))[1]

    def _score_grid(self, fb: np.ndarray, fa: np.ndarray, shift: np.float32, border: bool) -> np.ndarray:
        """The (G, G) score map as detect() builds it before upsampling."""
        c = self.cfg
        s = np.maximum(*change.direction_scores(fb, fa, c.radius)) - shift
        if border and c.border > 0:
            edge = np.ones_like(s, bool)
            edge[c.border:-c.border, c.border:-c.border] = False
            s = np.where(edge, np.minimum(s, 0), s).astype(np.float32)
        if c.blur > 0:
            s = cv2.GaussianBlur(s, (0, 0), c.blur)
        return s

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
        fb, fa = self._feats(before), self._feats(after)
        raw = np.maximum(*change.direction_scores(fb, fa, cfg.radius))
        shift = np.float32(np.median(raw))
        base_grid = self._score_grid(fb, fa, shift, border=True)
        base = cv2.resize(base_grid, (W, H), interpolation=cv2.INTER_LINEAR)
        windows = [Window(id=0, level=0, x0=0, y0=0, w=W, h=H, img_edges=(True, True, True, True))]
        grids = [base_grid]
        tiles, gated_out, by_level = [], 0, {0: 1}
        gated = cfg.tiles == "gated"
        for level in sorted(lv for lv in cfg.levels if lv > 0):
            for win in change.tile_windows(H, W, frac=TILE_FRAC[level], overlap=cfg.overlap):
                win.level = level
                y1, x1 = min(H, win.y0 + win.side), min(W, win.x0 + win.side)
                if gated and float(base[win.y0:y1, win.x0:x1].max()) < cfg.gate:
                    gated_out += 1                   # would add nothing: the gate keeps the whole-photo map there
                    continue
                sl = np.s_[win.y0:y1, win.x0:x1]
                tb, ta = self._feats(before[sl]), self._feats(after[sl])
                tiles.append((win, tb, ta))
                grids.append(self._score_grid(tb, ta, shift, border=False))
                windows.append(Window(id=len(windows), level=level, x0=win.x0, y0=win.y0, w=x1 - win.x0,
                                      h=y1 - win.y0, img_edges=(win.x0 == 0, win.y0 == 0, x1 >= W, y1 >= H)))
                by_level[level] = by_level.get(level, 0) + 1
        timings["stream"] = _ms(t0)

        t0 = time.perf_counter()
        res = change.detect(fb, fa, (H, W), self._change_cfg(), valid=valid, tiles=tiles, tile_mode="max",
                            gate=cfg.gate if gated else 0.0)
        dets = []
        for b in res.boxes:
            wid, level = self._strongest(b, windows, grids)
            dets.append(Detection(x1=b.x1, y1=b.y1, x2=b.x2, y2=b.y2, score=float(b.score), branch="after",
                                  window_id=wid, level=level, label=b.label))
        heat = np.clip(res.heat, 0, None) * np.float32(HEAT_AT_THR / cfg.thr)
        heat = cv2.resize(heat.astype(np.float32), (max(1, W // 4), max(1, H // 4)), interpolation=cv2.INTER_AREA)
        counts = {k: sum(d.label == k for d in dets) for k in ("appeared", "disappeared", "changed")}
        timings["postprocess"] = _ms(t0)
        timings["total"] = timings["load"] + _ms(t_start)
        meta = {
            "counts": counts, "windows_run": len(windows), "windows_by_level": {str(k): v for k, v in by_level.items()},
            "gated_out": gated_out, "pair": pair.pair_id, "backend": getattr(self.backend, "name", "?"),
            "work_size": [W, H], "scale": [float(v) for v in pair.meta.get("scale", (1.0, 1.0))],
        }
        return Result(pair_id=pair.pair_id, detections=dets, mask=np.asarray(res.mask, bool), heat=heat,
                      windows=windows, registration=reg, timings_ms={k: round(v, 3) for k, v in timings.items()},
                      before_aligned=before if cfg.register else None, meta=meta)

    @staticmethod
    def _strongest(b, windows: list[Window], grids: list[np.ndarray]) -> tuple[int, int]:
        """(window id, level) of the window whose score map peaks highest inside the box."""
        best, pick = -np.inf, (-1, 0)
        for win, g in zip(windows, grids):
            gh, gw = g.shape
            x1, y1 = max(b.x1, win.x0), max(b.y1, win.y0)
            x2, y2 = min(b.x2, win.x0 + win.w), min(b.y2, win.y0 + win.h)
            if x2 <= x1 or y2 <= y1:
                continue
            # a tile is square: its grid spans its full side even where the photo ends
            side_x = win.w if win.level == 0 else max(win.w, win.h)
            side_y = win.h if win.level == 0 else max(win.w, win.h)
            cx1, cx2 = int((x1 - win.x0) / side_x * gw), int(np.ceil((x2 - win.x0) / side_x * gw))
            cy1, cy2 = int((y1 - win.y0) / side_y * gh), int(np.ceil((y2 - win.y0) / side_y * gh))
            v = float(g[cy1:max(cy2, cy1 + 1), cx1:max(cx2, cx1 + 1)].max())
            if v > best:
                best, pick = v, (win.id, win.level)
        return pick
