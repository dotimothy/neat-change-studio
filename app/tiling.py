"""Window planning, coordinate mapping and the heat canvas.

CYWS sees a 256x256 input, so a 12 MP photo squashed to 256 loses every small
change. We therefore run a pyramid of windows over the canonical frame:

* level 0 - the whole image (or a few `short`-sided squares for very elongated
  images), which keeps the model in-distribution for large changes;
* level 1 - square tiles of side ``s1`` with overlap, for small changes;
* level 2 - optional half-size tiles, only when they are still >= 384 px.

Windows are always fully inside the image (last window flush with the edge, no
padding), so a crop never contains invented pixels.
"""

from __future__ import annotations

import math
from typing import Iterable, Literal, Optional, Sequence

import cv2
import numpy as np

from common import MODEL_SIZE, CywsError, Window

MIN_IMAGE_SIDE = 64       # below this the model sees almost nothing
MIN_TILE_SIDE = MODEL_SIZE  # never upscale a tile: that adds compute, not detail
LEVEL1_SKIP_FRAC = 0.8    # a tile this close to the full image adds nothing
LEVEL2_MIN_SIDE = 384
_EPS = 1e-9


# --------------------------------------------------------------------------- #
# Window planning
# --------------------------------------------------------------------------- #
def _spread(L: int, s: int, n: int) -> list[int]:
    """n positions of an s-wide window spread evenly over [0, L], last one flush."""
    if n <= 1:
        return [(L - s) // 2]
    return [int(round(i * (L - s) / (n - 1))) for i in range(n)]


def axis_positions(L: int, s: int, overlap: float) -> list[int]:
    """Start offsets of s-wide windows covering [0, L) with at least `overlap`.

    n = 1 if L <= s else ceil((L - s) / (s * (1 - overlap))) + 1, positions are
    round(i * (L - s) / (n - 1)), so the first window starts at 0 and the last
    one ends exactly at L. A single window is centred.
    """
    L, s = int(L), int(s)
    if s <= 0 or L <= 0:
        raise CywsError(f"axis length and window size must be positive (L={L}, s={s})")
    if s > L:
        raise CywsError(f"window size {s} exceeds axis length {L}")
    if not 0.0 <= overlap < 1.0:
        raise CywsError(f"overlap must be in [0, 1), got {overlap}")
    if L <= s:
        return _spread(L, s, 1)
    stride = s * (1.0 - overlap)
    n = math.ceil((L - s) / stride - _EPS) + 1
    return _spread(L, s, n)


def auto_tile_size(H: int, W: int, median_component_long_side: Optional[float] = None) -> int:
    """Level-1 tile side.

    With a hint (median change-component long side, in pixels) the tile is 4x
    that, so a typical change spans ~1/4 of the window like in CYWS training;
    otherwise half the short side. Clamped to [256, min(H, W)].
    """
    short = min(int(H), int(W))
    if median_component_long_side is not None:
        if not math.isfinite(median_component_long_side) or median_component_long_side <= 0:
            raise CywsError(
                f"median_component_long_side must be a positive number, got {median_component_long_side}"
            )
        s = int(round(4.0 * median_component_long_side))
    else:
        s = short // 2
    return int(min(max(s, MIN_TILE_SIDE), short))


def _edges(x0: int, y0: int, w: int, h: int, H: int, W: int) -> tuple[bool, bool, bool, bool]:
    return (x0 == 0, y0 == 0, x0 + w == W, y0 + h == H)


def _grid(H: int, W: int, s: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Row-major (x0, y0, w, h) squares of side s covering the image."""
    ys = axis_positions(H, s, overlap)
    xs = axis_positions(W, s, overlap)
    return [(x, y, s, s) for y in ys for x in xs]


def _level0(H: int, W: int, max_aspect_squash: float) -> list[tuple[int, int, int, int]]:
    short, long_ = min(H, W), max(H, W)
    if long_ / short <= max_aspect_squash + _EPS:
        return [(0, 0, W, H)]
    # Too elongated to squash: ceil(long/short) short-sided squares, evenly
    # spread so whatever overlap remains is shared equally.
    n = math.ceil(long_ / short - _EPS)
    pos = _spread(long_, short, n)
    if W >= H:
        return [(p, 0, short, short) for p in pos]
    return [(0, p, short, short) for p in pos]


def resolve_tile(tile: int | float | Literal["auto"], H: int, W: int) -> int | Literal["auto"]:
    """Turn a tile spec into pixels of an H x W frame.

    "auto" stays "auto" (min(H, W) // 2, see auto_tile_size); an int is pixels;
    a float in (0, 1] is a fraction of the short side, rounded, at least 64 px
    (so the default tile scales with the working resolution). plan_windows
    then clamps the side to [256, min(H, W)].
    """
    if tile == "auto":
        return "auto"
    if isinstance(tile, bool):
        raise CywsError(f"tile must be 'auto', a positive int or a fraction in (0, 1], got {tile!r}")
    if isinstance(tile, (int, np.integer)):
        if tile < 1:
            raise CywsError(f"tile must be 'auto', a positive int or a fraction in (0, 1], got {tile!r}")
        return int(tile)
    if isinstance(tile, (float, np.floating)) and 0.0 < float(tile) <= 1.0:
        return max(64, int(round(float(tile) * min(int(H), int(W)))))
    raise CywsError(f"tile must be 'auto', a positive int or a fraction in (0, 1], got {tile!r}")


def plan_windows(
    H: int,
    W: int,
    *,
    tile: int | float | Literal["auto"] = "auto",
    overlap: float = 0.25,
    levels: Sequence[int] = (0, 1),
    max_aspect_squash: float = 1.5,
    median_component_long_side: Optional[float] = None,
) -> list[Window]:
    """Plan the window pyramid for an H x W canonical frame.

    Ids are sequential with level 0 first, then level 1 and level 2, each in
    row-major order. Images whose short side is < 256 get level 0 only (it is
    upscaled to 256; tiling would only upscale further). If the requested
    levels would yield no window at all (e.g. levels=(1,) on an image where
    level 1 is skipped), level 0 is used so the image is still processed.
    """
    H, W = int(H), int(W)
    short = min(H, W)
    if short < MIN_IMAGE_SIDE:
        raise CywsError(f"image {W}x{H} is too small: short side must be >= {MIN_IMAGE_SIDE} px")
    req = tuple(int(lv) for lv in levels)
    if not req or any(lv not in (0, 1, 2) for lv in req):
        raise CywsError(f"levels must be a non-empty subset of (0, 1, 2), got {tuple(levels)}")
    if not 0.0 <= overlap < 1.0:
        raise CywsError(f"overlap must be in [0, 1), got {overlap}")
    if max_aspect_squash < 1.0:
        raise CywsError(f"max_aspect_squash must be >= 1, got {max_aspect_squash}")
    tile = resolve_tile(tile, H, W)

    plan: list[tuple[int, tuple[int, int, int, int]]] = []
    if 0 in req:
        plan += [(0, r) for r in _level0(H, W, max_aspect_squash)]
    if short >= MIN_TILE_SIDE:
        if tile == "auto":
            s1 = auto_tile_size(H, W, median_component_long_side)
        else:
            s1 = int(min(max(int(tile), MIN_TILE_SIDE), short))
        if 1 in req and s1 < LEVEL1_SKIP_FRAC * short:
            plan += [(1, r) for r in _grid(H, W, s1, overlap)]
        s2 = s1 // 2
        if 2 in req and s2 >= LEVEL2_MIN_SIDE:
            plan += [(2, r) for r in _grid(H, W, s2, overlap)]
    if not plan:
        plan = [(0, r) for r in _level0(H, W, max_aspect_squash)]

    plan.sort(key=lambda p: p[0])  # stable: keeps row-major order within a level
    return [
        Window(id=i, level=lv, x0=x, y0=y, w=w, h=h, img_edges=_edges(x, y, w, h, H, W))
        for i, (lv, (x, y, w, h)) in enumerate(plan)
    ]


# --------------------------------------------------------------------------- #
# Coordinate mapping
# --------------------------------------------------------------------------- #
def _as_boxes(boxes: np.ndarray) -> np.ndarray:
    arr = np.array(boxes, dtype=np.float64, copy=True)
    if arr.ndim not in (1, 2) or arr.shape[-1] < 4:
        raise CywsError(f"boxes must have shape (N, >=4) or (>=4,), got {arr.shape}")
    return arr


def window_to_canonical(boxes: np.ndarray, win: Window) -> np.ndarray:
    """Map boxes from the window's 256x256 model frame to the canonical frame.

    X = x0 + x * w / 256, Y = y0 + y * h / 256 on columns 0..3; any further
    columns (score, ...) are copied. Returns a new float64 array.
    """
    out = _as_boxes(boxes)
    sx, sy = win.w / MODEL_SIZE, win.h / MODEL_SIZE
    out[..., [0, 2]] = win.x0 + out[..., [0, 2]] * sx
    out[..., [1, 3]] = win.y0 + out[..., [1, 3]] * sy
    return out


def canonical_to_window(boxes: np.ndarray, win: Window) -> np.ndarray:
    """Inverse of window_to_canonical (canonical frame -> window's 256 frame)."""
    out = _as_boxes(boxes)
    sx, sy = MODEL_SIZE / win.w, MODEL_SIZE / win.h
    out[..., [0, 2]] = (out[..., [0, 2]] - win.x0) * sx
    out[..., [1, 3]] = (out[..., [1, 3]] - win.y0) * sy
    return out


def valid_fraction(win: Window, valid_mask: np.ndarray) -> float:
    """Fraction of the window's pixels that are True in the canonical valid mask."""
    if valid_mask.ndim != 2 or valid_mask.shape[0] < win.y1 or valid_mask.shape[1] < win.x1:
        raise CywsError(f"valid_mask of shape {valid_mask.shape} does not contain window {win}")
    crop = valid_mask[win.slices()]
    return float(np.count_nonzero(crop)) / crop.size if crop.size else 0.0


# --------------------------------------------------------------------------- #
# Heat canvas
# --------------------------------------------------------------------------- #
def _resize_axis(arr: np.ndarray, n: int, axis: int) -> np.ndarray:
    """Resize one axis to n: max-pool when shrinking (keeps peaks), linear when growing.

    Area/linear shrinking would average a narrow heat peak away, and the canvas
    is used for gating with an absolute threshold, so peaks must survive.
    """
    m = arr.shape[axis]
    if n == m:
        return arr
    if n < m:
        starts = (np.arange(n) * m) // n
        return np.maximum.reduceat(arr, starts, axis=axis)
    if axis == 0:
        return cv2.resize(arr, (arr.shape[1], n), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(arr, (n, arr.shape[0]), interpolation=cv2.INTER_LINEAR)


class HeatCanvas:
    """Per-pixel max of all window heatmaps on a 1/scale canonical grid."""

    def __init__(self, H: int, W: int, scale: int = 4) -> None:
        if H < 1 or W < 1 or scale < 1:
            raise CywsError(f"invalid heat canvas size H={H} W={W} scale={scale}")
        self.H, self.W, self.scale = int(H), int(W), int(scale)
        self.array = np.zeros((math.ceil(H / scale), math.ceil(W / scale)), dtype=np.float32)

    def footprint(self, win: Window) -> tuple[slice, slice]:
        """(rows, cols) canvas slices touched by the window (floor start, ceil end)."""
        s = self.scale
        r0, c0 = win.y0 // s, win.x0 // s
        r1, c1 = -(-win.y1 // s), -(-win.x1 // s)
        return slice(r0, max(r1, r0 + 1)), slice(c0, max(c1, c0 + 1))

    def add(self, win: Window, heat256: np.ndarray) -> None:
        """Resize the window's heat into its footprint and keep the per-pixel max."""
        if win.x0 < 0 or win.y0 < 0 or win.x1 > self.W or win.y1 > self.H:
            raise CywsError(f"window {win} lies outside the {self.W}x{self.H} canvas")
        heat = np.asarray(heat256, dtype=np.float32)
        if heat.ndim == 3 and heat.shape[0] == 1:
            heat = heat[0]
        if heat.ndim != 2 or heat.size == 0:
            raise CywsError(f"heat must be a 2-D map, got shape {np.shape(heat256)}")
        rows, cols = self.footprint(win)
        h = rows.stop - rows.start
        w = cols.stop - cols.start
        resized = _resize_axis(_resize_axis(np.ascontiguousarray(heat), h, 0), w, 1)
        dst = self.array[rows, cols]
        np.maximum(dst, resized, out=dst)

    def max_in(self, win: Window) -> float:
        """Highest heat any window has written inside this window's footprint."""
        return float(self.array[self.footprint(win)].max())


def gate_windows(
    windows: Iterable[Window],
    canvas: HeatCanvas,
    diff_energy: Optional[np.ndarray] = None,
    gate_heat: float = 0.1,
    diff_percentile: float = 75.0,
) -> list[Window]:
    """Keep the windows worth running at a finer level.

    A window is kept if the coarse heat inside it reaches `gate_heat`, or if its
    mean diff energy is strictly above the `diff_percentile`-th percentile of
    the per-window means (so a flat diff map keeps nothing extra). Input order
    is preserved.
    """
    wins = list(windows)
    if not wins:
        return []
    if not 0.0 <= diff_percentile <= 100.0:
        raise CywsError(f"diff_percentile must be in [0, 100], got {diff_percentile}")
    keep = np.array([canvas.max_in(w) >= gate_heat for w in wins], dtype=bool)
    if diff_energy is not None:
        energy = np.asarray(diff_energy, dtype=np.float32)
        if energy.shape != canvas.array.shape:
            raise CywsError(
                f"diff_energy shape {energy.shape} must match the heat canvas {canvas.array.shape}"
            )
        means = np.array([float(energy[canvas.footprint(w)].mean()) for w in wins])
        keep |= means > np.percentile(means, diff_percentile)
    return [w for w, k in zip(wins, keep) if k]
