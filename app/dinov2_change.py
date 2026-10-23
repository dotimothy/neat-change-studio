# Vendored from the DINOv2 project (MLSandbox/DINOv2/app/change.py), unchanged apart from this
# header. Keep it in step with that file; Change Studio wraps it in app/zeroshot.py.
"""Zero-shot change detection from DINOv2 patch features: no change-detection training at all.

Both photos go through the same feature extractor. A patch of the after photo that has no
similar patch nearby in the before photo is new (something appeared or changed there); a
patch of the before photo with no similar patch nearby in the after photo is gone. "Nearby"
is a (2r+1)^2 neighbourhood of patches, which forgives small camera shifts and parallax.

    score(p) = max( 1 - max_{q in N(p)} cos(after[p], before[q]),      # appeared / changed
                    1 - max_{q in N(p)} cos(before[p], after[q]) )     # disappeared

The per-patch score map is shifted by its median (so a global look change, light or white
balance, does not light up everything), upsampled to the photo, thresholded, cleaned and
split into connected regions; each region is a box with the region's peak score.

Inputs are photos in the same frame (the after photo's): resize the before photo to the
after's size, or register it first (registration.py from the CYWS app, optional).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class ChangeConfig:
    radius: int = 1                 # neighbourhood radius in patches (0 = same patch only)
    thr: float = 0.35               # change threshold on the median-shifted score
    median_shift: bool = True
    min_area: float = 0.0008        # smallest region, as a share of the photo
    blur: float = 0.0               # Gaussian blur of the score map, in patches (0 = off)
    open_px: int = 0                # morphological opening of the mask, in pixels of the working frame
    border: int = 0                 # ignore this many patches at the photo's edges (camera moved: edges come and go)
    label_radius: int = 2           # neighbourhood for telling appeared from disappeared (needs > 0)


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    area: float = 0.0               # region area as a share of the photo
    label: str = "changed"          # appeared, disappeared or changed (see label_for)


@dataclass
class ChangeResult:
    score_map: np.ndarray           # (G, G) float32, after the median shift
    heat: np.ndarray                # (H, W) float32 in the photo frame
    mask: np.ndarray                # (H, W) bool
    boxes: list[Box] = field(default_factory=list)


def resize_square(rgb_u8: np.ndarray, size: int) -> np.ndarray:
    """A photo squashed to size x size, as float32 RGB / 255 (the model input)."""
    interp = cv2.INTER_AREA if min(rgb_u8.shape[:2]) > size else cv2.INTER_LINEAR
    return cv2.resize(rgb_u8, (size, size), interpolation=interp).astype(np.float32) / 255.0


def normalize(patches: np.ndarray) -> np.ndarray:
    """L2-normalise the channel axis of (G, G, C) patch features."""
    n = np.linalg.norm(patches, axis=-1, keepdims=True)
    return patches / np.maximum(n, 1e-6)


def best_neighbour_similarity(a: np.ndarray, b: np.ndarray, radius: int) -> np.ndarray:
    """For each patch of a, the highest cosine to a patch of b within `radius` patches.
    a, b: (G, G, C) L2-normalised. Edges only compare against what exists."""
    g = a.shape[0]
    best = np.full(a.shape[:2], -1.0, np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ys, yd = slice(max(0, -dy), g - max(0, dy)), slice(max(0, dy), g - max(0, -dy))
            xs, xd = slice(max(0, -dx), g - max(0, dx)), slice(max(0, dx), g - max(0, -dx))
            sim = np.einsum("ijc,ijc->ij", a[ys, xs], b[yd, xd])
            best[ys, xs] = np.maximum(best[ys, xs], sim)
    return best


def direction_scores(before: np.ndarray, after: np.ndarray, radius: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """(appeared, gone), each (G, G): how unlike its neighbourhood in the other photo a patch of
    the after photo is (something new there), and a patch of the before photo (something gone)."""
    fb, fa = normalize(before), normalize(after)
    appeared = 1.0 - best_neighbour_similarity(fa, fb, radius)
    gone = 1.0 - best_neighbour_similarity(fb, fa, radius)
    return appeared.astype(np.float32), gone.astype(np.float32)


def change_scores(before: np.ndarray, after: np.ndarray, radius: int = 1, median_shift: bool = True) -> np.ndarray:
    """Per-patch change score (G, G) from two (G, G, C) patch maps."""
    appeared, gone = direction_scores(before, after, radius)
    s = np.maximum(appeared, gone)
    if median_shift:
        s = s - np.float32(np.median(s))
    return s


@dataclass
class Window:
    """A square crop of the photo, run through the model on its own: (x0, y0) and side, in pixels."""
    x0: int
    y0: int
    side: int
    level: int = 1


def tile_windows(h: int, w: int, frac: float = 0.6, overlap: float = 0.25) -> list[Window]:
    """Overlapping square tiles of side frac * short side, evenly spread, the last flush with
    the edge, so small objects are seen larger than in the whole-photo view."""
    side = max(1, round(frac * min(h, w)))

    def starts(length):
        if length <= side:
            return [0]
        n = int(np.ceil((length - side) / (side * (1 - overlap)))) + 1
        return [round(i * (length - side) / (n - 1)) for i in range(n)]
    return [Window(x, y, side) for y in starts(h) for x in starts(w)]


def detect(before: np.ndarray, after: np.ndarray, hw: tuple[int, int], cfg: ChangeConfig,
           valid: np.ndarray | None = None, tiles: list[tuple[Window, np.ndarray, np.ndarray]] = (),
           tile_mode: str = "max", gate: float = 0.0) -> ChangeResult:
    """Change mask and boxes in an (H, W) photo frame from two (G, G, C) patch maps of the
    whole photo, plus optional tiles: (window, before patches, after patches).

    Tile scores are shifted by the whole-photo median (a tile that is mostly change keeps its
    level) and pasted in; where tiles overlap, and against the whole-photo map, the higher
    score wins ("max") or the tile replaces it ("tile"). With gate > 0 a tile only counts
    where the whole-photo map is already above `gate` (fewer false alarms).
    `valid` (H, W) bool marks pixels the registered before photo covers; outside, no change."""
    h, w = hw
    raw = np.maximum(*direction_scores(before, after, cfg.radius))
    lr = max(cfg.label_radius, cfg.radius, 1)
    app_g, gone_g = direction_scores(before, after, lr)
    shift = np.float32(np.median(raw)) if cfg.median_shift else np.float32(0)
    s = raw - shift
    if cfg.border > 0:
        b = cfg.border
        edge = np.ones_like(s, bool)
        edge[b:-b, b:-b] = False
        s = np.where(edge, np.minimum(s, 0), s).astype(np.float32)
    if cfg.blur > 0:
        s = cv2.GaussianBlur(s, (0, 0), cfg.blur)
    heat = cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)
    # the two directions, kept apart only to label the boxes
    app_c = cv2.resize(app_g, (w, h), interpolation=cv2.INTER_LINEAR)
    gone_c = cv2.resize(gone_g, (w, h), interpolation=cv2.INTER_LINEAR)
    if tiles:
        base = heat.copy()
        tile_heat = np.full((h, w), -np.inf, np.float32)
        for win, fb, fa in tiles:
            t = np.maximum(*direction_scores(fb, fa, cfg.radius)) - shift
            ta, tg = direction_scores(fb, fa, lr)
            if cfg.blur > 0:
                t = cv2.GaussianBlur(t, (0, 0), cfg.blur)
            y1, x1 = min(h, win.y0 + win.side), min(w, win.x0 + win.side)
            fit = lambda m: cv2.resize(m, (win.side, win.side), interpolation=cv2.INTER_LINEAR)[: y1 - win.y0, : x1 - win.x0]
            region = tile_heat[win.y0:y1, win.x0:x1]
            np.maximum(region, fit(t), out=region)
            for canvas, m in ((app_c, ta), (gone_c, tg)):
                r = canvas[win.y0:y1, win.x0:x1]
                np.maximum(r, fit(m), out=r)
        covered = np.isfinite(tile_heat)
        if gate > 0:
            tile_heat = np.where(base >= gate, tile_heat, base)
        if tile_mode == "max":
            heat = np.where(covered, np.maximum(base, tile_heat), base)
        else:
            heat = np.where(covered, tile_heat, base)
        heat = heat.astype(np.float32)
    if valid is not None:
        heat = np.where(valid, heat, np.minimum(heat, 0)).astype(np.float32)
    mask = heat >= cfg.thr
    if cfg.open_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cfg.open_px + 1,) * 2)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    boxes, keep = [], np.zeros(n, bool)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        share = area / float(h * w)
        if share < cfg.min_area:
            continue
        keep[i] = True
        region = labels == i
        peak = float(heat[region].max())
        label = label_for(float(app_c[region].mean()), float(gone_c[region].mean()))
        boxes.append(Box(float(x), float(y), float(x + bw), float(y + bh), peak, share, label))
    mask = keep[labels]
    boxes.sort(key=lambda b: -b.score)
    return ChangeResult(score_map=s, heat=heat, mask=mask, boxes=boxes)


def label_for(appeared: float, gone: float, ratio: float = 1.25) -> str:
    """What kind of change a region is, from its mean appeared and gone scores. Something new
    (an object added) is unlike everything near it in the before photo, while the background it
    covers is still found next to it: appeared dominates. The reverse for something removed;
    both high for something replaced or moved."""
    if appeared > ratio * gone:
        return "appeared"
    if gone > ratio * appeared:
        return "disappeared"
    return "changed"


def overlay(after_u8: np.ndarray, res: ChangeResult, thr: float = 0.4, alpha: float = 0.55) -> np.ndarray:
    """The after photo with the change heat and the boxes drawn on it. RGB. The heat fades in
    from half the threshold, so only what is close to being called a change is tinted."""
    img = after_u8.copy()
    k = np.clip((res.heat - 0.5 * thr) / (0.5 * thr), 0, 1)
    colour = cv2.applyColorMap((80 + 175 * k).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
    a = (alpha * k)[..., None]
    img = (img * (1 - a) + colour * a).astype(np.uint8)
    t = max(2, round(max(img.shape[:2]) / 400))
    for b in res.boxes:
        cv2.rectangle(img, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (64, 224, 160), t)
        cv2.putText(img, f"{b.score:.2f}", (int(b.x1) + 3, int(b.y1) + 12 * t), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45 * t, (64, 224, 160), max(1, t // 2), cv2.LINE_AA)
    return img
