"""Shared types and contracts for the CYWS change-detection app.

Every module in app/ imports its data types from here, so the pieces agree on
one coordinate frame and one model I/O layout.

Coordinate frame
----------------
All boxes, windows and masks are in the *canonical frame*: the "after" image
(the query, I_q) after EXIF transposition, at the *working resolution*
(AppConfig.work_long: decoded with its long side >= work_long, and its short
side >= max(64, work_long // 2), by a power-of-two reduction, or native with
work_long = 0), in pixels, (x, y) with x to the
right, boxes as (x1, y1, x2, y2) floats with x2/y2 exclusive. The "before"
image (the reference, I_r) is resized into this frame, or warped into it when
registration is on. Pair.meta["scale"] = (sx, sy) converts canonical pixels
to original after-image pixels (x * sx, y * sy); outputs carry both.

Model I/O contract
------------------
Inputs (per inference, always together, always in this order):
    in0_before, in1_after : float32 (256, 256, 3) HWC, RGB, value = pixel / 255.
                            No mean/std normalisation (CYWS trains on /255).
Output (canonical, produced by every backend regardless of how the compiled
model lays it out):
    float32 (10, 256, 256) CHW with channels OUT_CHANNELS below.
    heat_* is already sigmoid-ed (0..1). wh_* is (w, h) in 256-pixel units.
    off_* is (dx, dy) sub-pixel centre offset in 256-pixel units.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np

MODEL_SIZE = 256  # CYWS input side (square)
INPUT_NAMES = ("in0_before", "in1_after")
OUT_CHANNELS = (
    "heat_b", "heat_a",
    "wh_b_w", "wh_b_h", "wh_a_w", "wh_a_h",
    "off_b_x", "off_b_y", "off_a_x", "off_a_y",
)
# Index helpers into the canonical (10, 256, 256) output.
CH = {name: i for i, name in enumerate(OUT_CHANNELS)}
BRANCH_SLICES = {
    # branch -> (heat channel, wh channels, offset channels)
    "before": (CH["heat_b"], (CH["wh_b_w"], CH["wh_b_h"]), (CH["off_b_x"], CH["off_b_y"])),
    "after": (CH["heat_a"], (CH["wh_a_w"], CH["wh_a_h"]), (CH["off_a_x"], CH["off_a_y"])),
}

Branch = Literal["before", "after"]
ChangeLabel = Literal["appeared", "disappeared", "changed"]

# One process-wide lock around every accelerator build / run / close.
# Building or closing a pyneat runner while another one is inside run() makes
# the running one fail with infra.accelerator_execution_failed and hang.
ACCELERATOR_LOCK = threading.RLock()


class CywsError(RuntimeError):
    """Raised for user-facing validation errors (bad paths, shapes, sizes)."""


@dataclass(frozen=True)
class PairItem:
    """A before/after pair on disk, before anything is decoded."""

    pair_id: str
    before_path: Path
    after_path: Path
    mask_path: Optional[Path] = None


@dataclass
class Pair:
    """A decoded pair in the canonical frame.

    after: uint8 (H, W, 3) RGB, defines the canonical frame.
    before: uint8 RGB. Same shape as after, except when the two aspect ratios
        differ by more than 2% and registration is on: then it keeps its own
        (EXIF-transposed) size and the app warps it into the canonical frame.
        meta["before_native"] is True in that case.
    gt_mask: bool (H, W) in the canonical frame, or None.
    meta: free-form provenance (original sizes, EXIF orientation, resize notes).
    """

    pair_id: str
    before: np.ndarray
    after: np.ndarray
    gt_mask: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Window:
    """A square-ish crop of the canonical frame that is fed to the model.

    img_edges: which window sides lie on the image border, (left, top, right, bottom).
    Boxes touching an interior (non-border) side at level >= 1 are truncated
    objects and get dropped, because an overlapping or coarser window sees them whole.
    """

    id: int
    level: int
    x0: int
    y0: int
    w: int
    h: int
    img_edges: tuple[bool, bool, bool, bool]

    @property
    def x1(self) -> int:
        return self.x0 + self.w

    @property
    def y1(self) -> int:
        return self.y0 + self.h

    def slices(self) -> tuple[slice, slice]:
        """(rows, cols) slices into an (H, W, ...) array."""
        return slice(self.y0, self.y1), slice(self.x0, self.x1)


@dataclass
class Detection:
    """One change box in the canonical frame."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    branch: Branch
    window_id: int = -1
    level: int = 0
    label: Optional[ChangeLabel] = None

    @property
    def box(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float64)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def to_dict(self, scale: Optional[tuple[float, float]] = None) -> dict:
        """JSON form; with scale = (sx, sy) also "box_orig", the box in original
        (native-resolution) after-image pixels: x * sx, y * sy."""
        d = {
            "box": [round(float(v), 2) for v in (self.x1, self.y1, self.x2, self.y2)],
            "score": round(float(self.score), 4),
            "branch": self.branch,
            "label": self.label,
            "window_id": self.window_id,
            "level": self.level,
        }
        if scale is not None:
            sx, sy = (float(v) for v in scale)
            d["box_orig"] = [round(float(v), 2) for v in (self.x1 * sx, self.y1 * sy, self.x2 * sx, self.y2 * sy)]
        return d


@dataclass
class RegResult:
    """Output of registration.register().

    H maps *before* pixel coordinates to *after* (canonical) coordinates,
    3x3 float64. valid_mask is bool (H, W) in the canonical frame, True where the
    warped before image has real pixels. method is "homography", "affine" or
    "identity"; ok is False when the estimate was rejected and identity used.
    """

    H: np.ndarray
    valid_mask: np.ndarray
    inliers: int
    inlier_ratio: float
    ok: bool
    method: str
    reason: str = ""


@dataclass
class AppConfig:
    """Every tunable of the pipeline; the CLI maps 1:1 onto these fields."""

    # Defaults = the "balanced" preset, tuned on the maintainer's private 28-pair
    # testCYWS set (FP32 host sweep, results/sweep_RESULTS.md): levels 0,1 with gated
    # tiles of 0.45 x the short side, thr 0.29, no registration.
    # tiling
    levels: tuple[int, ...] = (0, 1)
    tile: int | float | Literal["auto"] = 0.45   # px (int >= 64), fraction of the short side (float), or auto (= 0.5)
    overlap: float = 0.25
    tiles: Literal["full", "gated"] = "gated"
    gate_heat: float = 0.25                      # gated: run a level >= 1 window iff level-0 heat there >= this
    gate_diff_percentile: float = 75.0
    resize: Literal["area", "bilinear"] = "area"
    max_in_flight: int = 2
    # decode / CPU
    work_long: int = 1920              # canonical-frame long side floor (short side >= half); 0 = native
    preprocess_threads: int = 4        # crop workers overlapping the accelerator; 0 = inline
    # decode / filter / merge
    thr: float = 0.29
    topk: int = 100
    min_box_side: float = 6.0          # in 256-pixel units
    edge_margin: float = 2.0           # in 256-pixel units, truncation rule
    huge_box_frac: float = 0.7
    nms_iou: float = 0.5
    containment: float = 0.7
    merge: Literal["nms", "wbf"] = "nms"
    max_boxes: int = 50
    fuse_iou: float = 0.5
    # registration / false-positive filtering
    register: bool = False
    reg_method: Literal["sift", "orb"] = "sift"
    photometric: bool = False
    verify_change: Optional[bool] = None   # None -> on iff register
    verify_zncc: float = 0.85
    verify_diff_ratio: float = 1.5
    min_valid_frac: float = 0.5
    # masks / output
    mask_mode: Literal["box", "diff"] = "box"
    allow_squash: bool = False
    save_heat: bool = False

    def verify_enabled(self) -> bool:
        return self.register if self.verify_change is None else bool(self.verify_change)


# Named configurations (AppConfig field overrides); the CLI's --preset. "balanced"
# is exactly the AppConfig defaults. Pooled numbers on testCYWS (28 pairs, FP32,
# work_long 1920): fast F1 0.590 / 30 of 47 GT components / 2 of 15 no-change pairs
# with boxes / 1.1 windows per pair; balanced 0.594 / 34 / 2 / 3.5; accurate 0.591 /
# 34 / 2 / 13.3. accurate runs every level-1 tile (no gating), so it can still find
# a change that leaves no heat at level 0.
PRESETS: dict[str, dict] = {
    "fast": dict(levels=(0,), tile=0.45, tiles="gated", gate_heat=0.25, thr=0.29, register=False),
    "balanced": dict(levels=(0, 1), tile=0.45, tiles="gated", gate_heat=0.25, thr=0.29, register=False),
    "accurate": dict(levels=(0, 1), tile=0.45, tiles="full", gate_heat=0.25, thr=0.29, register=False),
}


def preset_config(name: str = "balanced", **overrides) -> "AppConfig":
    """AppConfig for a named preset, with any field overridden by keyword."""
    if name not in PRESETS:
        raise CywsError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return AppConfig(**{**PRESETS[name], **overrides})


@dataclass
class Result:
    """Everything CywsApp.process() produces for one pair."""

    pair_id: str
    detections: list[Detection]
    mask: np.ndarray                      # bool (H, W), canonical frame
    heat: np.ndarray                      # float32 (H//4, W//4) max heat canvas
    windows: list[Window]
    registration: Optional[RegResult]
    timings_ms: dict[str, float]
    before_aligned: Optional[np.ndarray] = None  # uint8 (H, W, 3) in canonical frame
    meta: dict = field(default_factory=dict)
