"""Decode images into the canonical frame and turn windows into model tensors.

The after image (EXIF-transposed) defines the canonical frame. Everything that
must agree with it - the before image, the GT mask - is reconciled here once,
so later stages never have to think about orientation or sizes again.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, Optional, Union

import cv2
import numpy as np
from PIL import Image, ImageOps

from common import MODEL_SIZE, CywsError, Pair, PairItem, Window

PathLike = Union[str, Path]
Interp = Literal["area", "bilinear"]

MIN_SIDE = 64              # smaller images cannot be meaningfully tiled
ASPECT_TOL = 0.02          # relative aspect difference treated as "same shape"
MASK_MIN_COMPONENT_FRAC = 1e-4  # drop GT specks below 0.01 % of the image
EXIF_ORIENTATION_TAG = 0x0112
_HIGH_BIT_MODES = ("I;16", "I;16L", "I;16B", "I;16N", "I", "F")
JPEG_FORMATS = ("JPEG", "MPO")  # PIL formats whose decoder supports draft (DCT-domain) scaling


def aspect_close(wh_a: tuple[int, int], wh_b: tuple[int, int], tol: float = ASPECT_TOL) -> bool:
    """True if two (w, h) sizes have aspect ratios within `tol` (symmetric)."""
    ra = wh_a[0] / wh_a[1]
    rb = wh_b[0] / wh_b[1]
    return max(ra, rb) / min(ra, rb) - 1.0 <= tol + 1e-12


def apply_exif_orientation(arr: np.ndarray, orientation: int) -> np.ndarray:
    """Apply EXIF orientation 1..8 to an (H, W[, C]) array, like ImageOps.exif_transpose.

    Needed for masks, which are stored in the after image's raw pixel frame
    but carry no EXIF of their own.
    """
    if orientation not in range(1, 9):
        raise CywsError(f"EXIF orientation must be 1..8, got {orientation}")
    ops = {
        1: lambda a: a,
        2: lambda a: a[:, ::-1],                    # mirror horizontal
        3: lambda a: a[::-1, ::-1],                 # rotate 180
        4: lambda a: a[::-1],                       # mirror vertical
        5: lambda a: np.swapaxes(a, 0, 1),          # transpose
        6: lambda a: np.rot90(a, k=-1),             # rotate 90 clockwise
        7: lambda a: np.swapaxes(a, 0, 1)[::-1, ::-1],  # transverse
        8: lambda a: np.rot90(a, k=1),              # rotate 90 counter-clockwise
    }
    return np.ascontiguousarray(ops[orientation](arr))


def short_floor_for(work_long: int) -> int:
    """Smallest short side a reduction may leave: max(MIN_SIDE, work_long // 2).

    Never binds for aspect ratios up to 2:1 (4:3, 16:9 ...), where the long-side
    rule alone decides; for panoramas and strips it stops the reduction from
    starving the short side (and the level-1 tiling planned on it).
    """
    return max(MIN_SIDE, int(work_long) // 2)


def reduction_for(long_side: int, work_long: int, max_factor: int = 0, *, short_side: Optional[int] = None) -> int:
    """Largest power of two s >= 1 with long_side // s >= work_long (1 if work_long <= 0).

    short_side, when given, must also stay >= short_floor_for(work_long) after
    the reduction, so elongated images keep their short-side resolution.
    max_factor > 0 caps s (JPEG draft decoding can only reduce by 1/2, 1/4, 1/8).
    """
    s = 1
    if work_long <= 0:
        return s
    floor = short_floor_for(work_long)
    while (
        long_side // (2 * s) >= work_long
        and (short_side is None or int(short_side) // (2 * s) >= floor)
        and (max_factor <= 0 or 2 * s <= max_factor)
    ):
        s *= 2
    return s


def _open_image(path: PathLike, what: str, work_long: int = 0) -> tuple[Path, Image.Image, tuple[int, int]]:
    """Open and fully decode an image so truncated files fail here, not later.

    With work_long > 0 a JPEG is decoded in PIL draft mode: the decoder scales
    the DCT blocks by 1/2, 1/4 or 1/8 - the largest reduction that keeps the
    long side >= work_long and the short side >= short_floor_for(work_long) -
    which is several times faster than a full decode
    followed by a resize. Returns (path, image, raw size before any reduction).
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise CywsError(f"{what} does not exist: {p}")
    if not p.is_file():
        raise CywsError(f"{what} is not a file: {p}")
    try:
        img = Image.open(p)
        raw_size = (int(img.width), int(img.height))
        if work_long > 0 and img.format in JPEG_FORMATS:
            s = reduction_for(max(raw_size), int(work_long), max_factor=8, short_side=min(raw_size))
            if s > 1:  # implies min(raw_size) // s >= MIN_SIDE, so the requested size has no 0
                # Requesting (w // s, h // s) makes draft pick exactly 1/s:
                # w // (w // s) >= s, and < 2s whenever w // s >= 2.
                img.draft("RGB", (raw_size[0] // s, raw_size[1] // s))
        img.load()
    except (OSError, ValueError, SyntaxError, ZeroDivisionError, Image.DecompressionBombError) as exc:
        raise CywsError(f"cannot read {what} {p}: {exc}") from exc
    return p, img, raw_size


def _exif_orientation(img: Image.Image) -> int:
    try:
        value = int(img.getexif().get(EXIF_ORIENTATION_TAG, 1))
    except (TypeError, ValueError, OSError):
        return 1
    return value if 1 <= value <= 8 else 1


def _exif_transpose(img: Image.Image, p: Path, what: str) -> Image.Image:
    try:
        out = ImageOps.exif_transpose(img)
    except (OSError, ValueError, SyntaxError) as exc:
        raise CywsError(f"cannot apply EXIF orientation of {what} {p}: {exc}") from exc
    return img if out is None else out


def _high_bit_to_u8(img: Image.Image) -> np.ndarray:
    """Scale 16-bit / int32 / float single-channel images into 0..255."""
    a = np.asarray(img, dtype=np.float64)
    if img.mode.startswith("I;16"):
        a = a / 257.0
    else:
        hi = float(a.max()) if a.size else 0.0
        if hi > 255.0:
            a = a * (255.0 / hi)
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def _to_gray_u8(img: Image.Image) -> np.ndarray:
    if img.mode in _HIGH_BIT_MODES:
        return _high_bit_to_u8(img)
    return np.asarray(img.convert("L"), dtype=np.uint8)


def _to_rgb_u8(img: Image.Image) -> np.ndarray:
    if img.mode in _HIGH_BIT_MODES:
        gray = _high_bit_to_u8(img)
        return np.ascontiguousarray(np.repeat(gray[:, :, None], 3, axis=2))
    rgb = img if img.mode == "RGB" else img.convert("RGB")
    return np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))


_DECODE_POOL: Optional[ThreadPoolExecutor] = None
_DECODE_POOL_LOCK = threading.Lock()


def _decode_pool() -> ThreadPoolExecutor:
    """Shared workers for load_pair: the before image, the after image and the GT
    mask decode at the same time (PIL and cv2 release the GIL while decoding)."""
    global _DECODE_POOL
    with _DECODE_POOL_LOCK:
        if _DECODE_POOL is None:
            _DECODE_POOL = ThreadPoolExecutor(max_workers=3, thread_name_prefix="cyws-decode")
        return _DECODE_POOL


def _peek(path: PathLike, work_long: int) -> Optional[dict]:
    """What read_image_rgb(path, work_long) will produce, from the header only.

    Returns raw_size, exif_orientation, the working long side and the working
    (H, W) after the EXIF transpose, or None when
    the header cannot be read (the real decode then reports the error).
    """
    try:
        with Image.open(Path(path).expanduser()) as img:
            raw = (int(img.width), int(img.height))
            orientation = _exif_orientation(img)
            if work_long > 0 and img.format in JPEG_FORMATS:
                s = reduction_for(max(raw), int(work_long), max_factor=8, short_side=min(raw))
                if s > 1:
                    img.draft("RGB", (raw[0] // s, raw[1] // s))  # sets the size; decodes nothing
            w, h = int(img.width), int(img.height)
    except Exception:  # noqa: BLE001 - a bad file fails properly in read_image_rgb
        return None
    extra = reduction_for(max(w, h), int(work_long), short_side=min(w, h)) if work_long > 0 else 1
    if extra > 1:
        w, h = -(-w // extra), -(-h // extra)
    hw = (w, h) if orientation in (5, 6, 7, 8) else (h, w)  # (H, W) after the EXIF transpose
    return {"raw_size": raw, "exif_orientation": orientation, "long": max(w, h), "work_hw": hw}


def read_image_rgb(path: PathLike, work_long: int = 0) -> tuple[np.ndarray, dict]:
    """Decode an image as EXIF-transposed RGB uint8 (H, W, 3).

    work_long > 0 decodes at reduced resolution with the long side kept
    >= work_long and the short side >= short_floor_for(work_long) (= max(64,
    work_long // 2); binds only above 2:1 aspect): JPEGs by draft (DCT-domain 1/2, 1/4, 1/8) decoding, anything
    else (and a JPEG still >= 2 * work_long after draft) by a power-of-two
    INTER_AREA downscale, applied only when the long side is >= 2 * work_long.
    work_long = 0 decodes at native resolution.

    meta: raw_size (w, h stored, before transpose and reduction), full_size
    (w, h after transpose, native resolution), size (w, h of the returned
    array), scale (full / returned, x and y), reduction (total power-of-two
    factor), draft (factor done by the JPEG decoder), exif_orientation (1..8),
    mode (original PIL mode), path.
    """
    work_long = int(work_long)
    if work_long < 0:
        raise CywsError(f"work_long must be >= 0, got {work_long}")
    p, img, raw_size = _open_image(path, "image", work_long)
    orientation = _exif_orientation(img)
    mode = img.mode
    draft = max(1, int(round(max(raw_size) / max(img.width, img.height))))
    img = _exif_transpose(img, p, "image")
    rgb = _to_rgb_u8(img)
    full_w, full_h = (raw_size[1], raw_size[0]) if orientation in (5, 6, 7, 8) else raw_size
    extra = reduction_for(max(rgb.shape[:2]), work_long, short_side=min(rgb.shape[:2]))
    if extra > 1:
        h0, w0 = rgb.shape[:2]
        rgb = cv2.resize(rgb, (-(-w0 // extra), -(-h0 // extra)), interpolation=cv2.INTER_AREA)
    h, w = rgb.shape[:2]
    if min(full_w, full_h) < MIN_SIDE:
        raise CywsError(f"image {p} is {full_w}x{full_h}; both sides must be >= {MIN_SIDE} px")
    if min(h, w) < MIN_SIDE:
        raise CywsError(f"image {p} decodes to {w}x{h} at work_long={work_long}; both sides must be >= {MIN_SIDE} px")
    meta = {
        "path": str(p),
        "raw_size": raw_size,
        "full_size": (int(full_w), int(full_h)),
        "size": (int(w), int(h)),
        "scale": (full_w / w, full_h / h),
        "reduction": int(draft * extra),
        "draft": int(draft),
        "exif_orientation": orientation,
        "mode": mode,
    }
    return np.ascontiguousarray(rgb), meta


def read_mask(
    path: PathLike,
    target_hw: tuple[int, int],
    raw_after_wh: tuple[int, int],
    after_orientation: int,
    out_hw: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """Load a GT change mask as bool (H, W) in the canonical frame.

    Masks are matched against the after image: at its raw (untransposed) size
    they get the after image's EXIF transpose; at the canonical size they are
    used as is; at another size with the canonical (or raw) aspect within 2 %
    they are nearest-resized (then transposed if raw). A mask's own EXIF
    orientation, if any, is honoured first. JPEG masks are thresholded at > 127,
    opened with a 3x3 kernel and stripped of components below 0.01 % of the image.

    target_hw is the after image's native (EXIF-transposed) size; all of the
    reconciliation above happens there. out_hw, when given and different,
    is the working frame the cleaned mask is then nearest-resized into.
    """
    H, W = (int(v) for v in target_hw)
    rw, rh = (int(v) for v in raw_after_wh)
    if min(H, W, rw, rh) <= 0:
        raise CywsError(f"invalid mask target size {target_hw} / raw size {raw_after_wh}")
    if after_orientation not in range(1, 9):
        raise CywsError(f"after_orientation must be 1..8, got {after_orientation}")

    p, img, _ = _open_image(path, "mask")
    if _exif_orientation(img) != 1:
        img = _exif_transpose(img, p, "mask")
    gray = _to_gray_u8(img)
    mh, mw = gray.shape
    mask_wh = (mw, mh)

    if mask_wh == (rw, rh):
        gray = apply_exif_orientation(gray, after_orientation)
    elif mask_wh == (W, H):
        pass
    elif aspect_close(mask_wh, (W, H)):
        gray = cv2.resize(gray, (W, H), interpolation=cv2.INTER_NEAREST)
    elif aspect_close(mask_wh, (rw, rh)):
        gray = cv2.resize(gray, (rw, rh), interpolation=cv2.INTER_NEAREST)
        gray = apply_exif_orientation(gray, after_orientation)
    else:
        raise CywsError(
            f"mask {p} is {mw}x{mh}, which matches neither the after image "
            f"({W}x{H}, raw {rw}x{rh}) nor its aspect ratio"
        )
    if gray.shape != (H, W):
        raise CywsError(
            f"mask {p} ended up {gray.shape[1]}x{gray.shape[0]} instead of {W}x{H}; "
            "raw size and orientation do not agree with the target size"
        )

    binary = (gray > 127).astype(np.uint8)
    if out_hw is not None:
        oh, ow = (int(v) for v in out_hw)
        if min(oh, ow) <= 0:
            raise CywsError(f"invalid mask output size {out_hw}")
        if (oh, ow) != (H, W):
            # Clean up at the working resolution, not the native one: on a 12 MP
            # mask the opening and labelling below cost ~250 ms on the board's
            # A65. A pixel is set when most of the native pixels it covers are.
            shrinking = oh < H and ow < W
            binary = cv2.resize(binary * np.uint8(255), (ow, oh),
                                interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_NEAREST)
            binary = (binary > 127).astype(np.uint8)
            H, W = oh, ow
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= MASK_MIN_COMPONENT_FRAC * H * W
    keep[0] = False  # background label
    return keep[labels] if n > 1 else np.zeros((H, W), dtype=bool)


def _interp_flag(src_wh: tuple[int, int], dst_wh: tuple[int, int]) -> int:
    """INTER_AREA when the pixel count shrinks (anti-aliasing), else INTER_LINEAR."""
    return cv2.INTER_AREA if dst_wh[0] * dst_wh[1] < src_wh[0] * src_wh[1] else cv2.INTER_LINEAR


def resize_to(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize to (w, h): INTER_AREA if shrinking, else INTER_LINEAR.

    Returns the input unchanged when it already has that size.
    """
    if not isinstance(img, np.ndarray) or img.ndim not in (2, 3) or img.size == 0:
        raise CywsError("resize_to expects a non-empty (H, W) or (H, W, C) array")
    w, h = int(w), int(h)
    if w <= 0 or h <= 0:
        raise CywsError(f"resize target must be positive, got {w}x{h}")
    src_h, src_w = img.shape[:2]
    if (src_w, src_h) == (w, h):
        return img
    return cv2.resize(img, (w, h), interpolation=_interp_flag((src_w, src_h), (w, h)))


def load_pair(
    item: PairItem, *, allow_squash: bool = False, register: bool = False, work_long: int = 0
) -> Pair:
    """Decode a PairItem into a Pair in the canonical (after) frame.

    work_long > 0 makes the canonical frame a reduced-resolution "working"
    frame: the after image is decoded with its long side >= work_long (see
    read_image_rgb), and the before image is decoded with its long side >= the
    after's working long side, so it is never upscaled more than its native
    size forces. meta["scale"] = (sx, sy) original after pixels per working
    pixel, meta["orig_size"] = (w, h) of the EXIF-transposed after image at
    native resolution, meta["work_size"] = (w, h) of the canonical frame.

    The before image is resized to the after size when the aspect ratios agree
    within 2 %. Otherwise registration needs the native geometry (kept, with
    meta["before_native"] = True); squashing is opt-in because it distorts
    every object; and with neither flag the pair is rejected. A GT mask is
    reconciled at the after image's native size and nearest-resized into the
    canonical frame.
    """
    work_long = int(work_long)
    if work_long < 0:
        raise CywsError(f"work_long must be >= 0, got {work_long}")
    # The before image is decoded at the after image's working long side, and the
    # mask needs the after image's raw size and orientation. All of that is in the
    # JPEG headers, so the three decodes run at the same time.
    pool = _decode_pool()
    peek = _peek(item.after_path, work_long)
    before_long = (peek["long"] if peek else None) if work_long > 0 else 0
    f_after = pool.submit(read_image_rgb, item.after_path, work_long)
    f_before = pool.submit(read_image_rgb, item.before_path, before_long) if before_long is not None else None
    f_mask = None
    if item.mask_path is not None and peek is not None:
        f_mask = pool.submit(_timed_mask, item.mask_path, peek["raw_size"], peek["exif_orientation"],
                             peek["work_hw"])
    try:
        after, after_meta = f_after.result()
    except BaseException:
        for f in (f_before, f_mask):
            if f is not None:
                f.cancel()
        raise
    H, W = after.shape[:2]
    if f_before is not None and before_long != (max(H, W) if work_long > 0 else 0):
        f_before.cancel()  # header prediction was off (should not happen): decode again, serially
        f_before = None
    if f_before is None:
        before, before_meta = read_image_rgb(item.before_path, max(H, W) if work_long > 0 else 0)
    else:
        before, before_meta = f_before.result()
    bh, bw = before.shape[:2]
    full_w, full_h = after_meta["full_size"]
    meta: dict = {
        "after": after_meta,
        "before": before_meta,
        "work_long": work_long,
        "orig_size": (int(full_w), int(full_h)),
        "work_size": (int(W), int(H)),
        "scale": (full_w / W, full_h / H),
        "before_native": False,
        "before_resized": False,
        "squashed": False,
        "warnings": [],
        "mask": None,
    }

    if aspect_close((bw, bh), (W, H)):
        if (bw, bh) != (W, H):
            before = resize_to(before, W, H)
            meta["before_resized"] = True
    elif register:
        meta["before_native"] = True
    elif allow_squash:
        before = resize_to(before, W, H)
        meta["before_resized"] = True
        meta["squashed"] = True
        meta["warnings"].append(
            f"before {bw}x{bh} squashed to after {W}x{H} (aspect {bw / bh:.3f} vs "
            f"{W / H:.3f}); objects are distorted - prefer --register"
        )
    else:
        raise CywsError(
            f"pair '{item.pair_id}': before is {bw}x{bh} (aspect {bw / bh:.3f}) but after is "
            f"{W}x{H} (aspect {W / H:.3f}); differ by more than {ASPECT_TOL:.0%}. "
            "Use --register to align them or --allow-squash to stretch the before image"
        )

    gt_mask = None
    if item.mask_path is not None:
        same_header = (f_mask is not None and peek["raw_size"] == tuple(after_meta["raw_size"])
                       and peek["exif_orientation"] == after_meta["exif_orientation"])
        if same_header:
            gt_mask, mask_ms = f_mask.result()
            if gt_mask.shape != (H, W):
                same_header = False
        if not same_header:
            gt_mask, mask_ms = _timed_mask(
                item.mask_path, after_meta["raw_size"], after_meta["exif_orientation"], (H, W)
            )
        meta["mask"] = {
            "path": str(item.mask_path),
            "positive_frac": float(gt_mask.mean()),
            # GT only. Decoded alongside the images, so it mostly overlaps them in the load time.
            "read_ms": round(mask_ms, 3),
        }

    return Pair(pair_id=item.pair_id, before=before, after=after, gt_mask=gt_mask, meta=meta)


def _timed_mask(
    path: PathLike, raw_after_wh: tuple[int, int], after_orientation: int, out_hw: tuple[int, int],
) -> tuple[np.ndarray, float]:
    """read_mask reconciled against the after image's native frame, returned at
    out_hw (the working frame), with its time in ms."""
    t0 = time.perf_counter()
    rw, rh = raw_after_wh
    full_w, full_h = (rh, rw) if after_orientation in (5, 6, 7, 8) else (rw, rh)
    mask = read_mask(path, (full_h, full_w), (rw, rh), after_orientation, out_hw=out_hw)
    return mask, (time.perf_counter() - t0) * 1e3


def _check_window(img: np.ndarray, win: Window, name: str) -> None:
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        shape = getattr(img, "shape", None)
        dtype = getattr(img, "dtype", None)
        raise CywsError(f"{name} must be uint8 (H, W, 3), got {dtype} {shape}")
    H, W = img.shape[:2]
    if win.w <= 0 or win.h <= 0 or win.x0 < 0 or win.y0 < 0 or win.x1 > W or win.y1 > H:
        raise CywsError(
            f"window {win.id} ({win.x0},{win.y0},{win.w}x{win.h}) is outside the {W}x{H} {name}"
        )


def preprocess(img_u8: np.ndarray, win: Window, interp: Interp = "area") -> np.ndarray:
    """Crop a window and produce the model input: float32 (256, 256, 3) RGB / 255.

    "area" uses INTER_AREA when shrinking (no aliasing on big tiles) and
    INTER_LINEAR when growing; "bilinear" always uses INTER_LINEAR.
    """
    if interp not in ("area", "bilinear"):
        raise CywsError(f"interp must be 'area' or 'bilinear', got {interp!r}")
    _check_window(img_u8, win, "image")
    crop = img_u8[win.slices()]
    size = (MODEL_SIZE, MODEL_SIZE)
    if (win.w, win.h) == size:
        resized = crop
    else:
        flag = _interp_flag((win.w, win.h), size) if interp == "area" else cv2.INTER_LINEAR
        resized = cv2.resize(crop, size, interpolation=flag)
    out = np.ascontiguousarray(resized, dtype=np.float32)  # always a new array (uint8 in)
    out *= np.float32(1.0 / 255.0)
    return out


def crop_pair(
    before: np.ndarray, after: np.ndarray, win: Window, interp: Interp = "area"
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess the same window of both images, so the two tensors stay aligned."""
    b_shape = getattr(before, "shape", None)
    a_shape = getattr(after, "shape", None)
    if b_shape != a_shape:
        raise CywsError(
            f"before {b_shape} and after {a_shape} must have identical shapes; "
            "resize or warp the before image into the canonical frame first"
        )
    return preprocess(before, win, interp), preprocess(after, win, interp)
