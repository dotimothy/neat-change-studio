"""Before -> after registration and change verification.

CYWS compares the two images pixel-for-pixel inside each window, so any camera
motion between the shots shows up as "change". This module estimates a
homography from the before image into the canonical (after) frame, warps the
before image with it, and offers a cheap per-box check that drops detections
which are only explained by a small residual shift (parallax, misregistration).

All coordinates use OpenCV's pixel-centre convention, which is what keypoints,
``findHomography`` and ``warpPerspective`` share; the work-scale <-> full-res
scale matrices below include the half-pixel term so the lifted homography is
exact rather than off by up to half a work pixel.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from common import CywsError, Detection, RegResult

# Acceptance rules for an estimated transform (shared by homography and affine).
MIN_INLIERS = 30
MIN_INLIER_RATIO = 0.25
DET_RANGE = (0.5, 2.0)
MAX_PERSPECTIVE = 2e-3          # |h31|, |h32| at work scale, H normalised to h33 = 1
RATIO_TEST = 0.75
REPROJ_THR = 3.0                # px at work scale
# Seed for cv2's RNG (USAC/RANSAC sampling, FLANN trees) at the start of every
# register() call: otherwise H depends on what ran before in the thread (entries
# differed by up to 0.3 between calls), and so do the boxes.
RNG_SEED = 12345
VALID_ERODE_PX = 2
SIFT_FEATURES = 4000
ORB_FEATURES = 5000
MIN_IMAGE_SIDE = 16

# verify_detection
MIN_VERIFY_SIDE = 8             # boxes smaller than this are always kept
VERIFY_MAX_SIDE = 256           # coarse-to-fine search above this box side
FLAT_STD = 2.0                  # gray-level std below which a patch has no structure
MIN_GLOBAL_DIFF = 1.0           # floor for global_median_diff (identical images)


# --------------------------------------------------------------------------- helpers


def _check_image(img: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(img, np.ndarray):
        raise CywsError(f"{name}: expected a numpy array, got {type(img).__name__}")
    if img.dtype != np.uint8:
        raise CywsError(f"{name}: expected uint8, got {img.dtype}")
    if img.ndim == 3 and img.shape[2] == 3:
        pass
    elif img.ndim != 2:
        raise CywsError(f"{name}: expected (H, W, 3) or (H, W), got shape {img.shape}")
    if min(img.shape[:2]) < MIN_IMAGE_SIDE:
        raise CywsError(f"{name}: image too small {img.shape[:2]} (min side {MIN_IMAGE_SIDE})")
    return img


def _gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def _scale_matrix(sx: float, sy: float) -> np.ndarray:
    """Pixel-centre scale: maps (x, y) to (sx*x + (sx-1)/2, sy*y + (sy-1)/2)."""
    return np.array(
        [[sx, 0.0, 0.5 * (sx - 1.0)], [0.0, sy, 0.5 * (sy - 1.0)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _to_work(gray: np.ndarray, work_long: int) -> tuple[np.ndarray, np.ndarray]:
    """Downscale to long side work_long (never upscale); returns (image, full->work S)."""
    h, w = gray.shape[:2]
    s = min(1.0, work_long / float(max(h, w)))
    if s >= 1.0:
        return gray, np.eye(3)
    ww, wh = max(1, round(w * s)), max(1, round(h * s))
    small = cv2.resize(gray, (ww, wh), interpolation=cv2.INTER_AREA)
    return small, _scale_matrix(ww / w, wh / h)


def _detector(method: str):
    """SIFT unless ORB is requested or this cv2 build lacks SIFT."""
    if method == "sift":
        try:
            return cv2.SIFT_create(nfeatures=SIFT_FEATURES), "sift"
        except (AttributeError, cv2.error):
            pass
    return cv2.ORB_create(nfeatures=ORB_FEATURES), "orb"


def _match(des_b: np.ndarray, des_a: np.ndarray, kind: str) -> list:
    """kNN (k=2) matches from before to after, filtered by Lowe's ratio test."""
    if des_b is None or des_a is None or len(des_b) < 2 or len(des_a) < 2:
        return []
    if kind == "sift":
        try:
            matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 64})
            knn = matcher.knnMatch(des_b, des_a, k=2)
        except cv2.error:
            knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_b, des_a, k=2)
    else:
        knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des_b, des_a, k=2)
    return [p[0] for p in knn if len(p) == 2 and p[0].distance < RATIO_TEST * p[1].distance]


def _find_homography(src: np.ndarray, dst: np.ndarray) -> tuple[Optional[np.ndarray], np.ndarray]:
    method = getattr(cv2, "USAC_MAGSAC", None)
    H, mask = (None, None)
    if method is not None:
        try:
            H, mask = cv2.findHomography(src, dst, method, REPROJ_THR, maxIters=5000, confidence=0.999)
        except cv2.error:
            H = None
    if H is None:
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, REPROJ_THR, maxIters=5000, confidence=0.999)
    inl = np.zeros(len(src), bool) if mask is None else mask.ravel().astype(bool)
    return H, inl


def _corners(w: int, h: int) -> np.ndarray:
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)


def _shape_reason(H: np.ndarray, w: int, h: int) -> str:
    """Empty string if H is a plausible camera motion of a w x h image, else why not."""
    if not np.all(np.isfinite(H)) or abs(H[2, 2]) < 1e-12:
        return "degenerate matrix"
    H = H / H[2, 2]
    det = float(np.linalg.det(H[:2, :2]))
    if not DET_RANGE[0] <= det <= DET_RANGE[1]:
        return f"determinant {det:.3f} outside {DET_RANGE}"
    if max(abs(H[2, 0]), abs(H[2, 1])) >= MAX_PERSPECTIVE:
        return f"perspective terms ({H[2, 0]:.2e}, {H[2, 1]:.2e}) too large"
    pts = _corners(w, h)
    den = H[2, 0] * pts[:, 0] + H[2, 1] * pts[:, 1] + 1.0
    if np.any(den <= 0):
        return "corners map through the horizon"
    warped = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    edges = np.roll(warped, -1, axis=0) - warped
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    if np.any(cross <= 0):
        return "warped corners not convex or flipped"
    return ""


def _inlier_reason(n_in: int, ratio: float) -> str:
    if n_in < MIN_INLIERS:
        return f"{n_in} inliers < {MIN_INLIERS}"
    if ratio < MIN_INLIER_RATIO:
        return f"inlier ratio {ratio:.2f} < {MIN_INLIER_RATIO}"
    return ""


def _valid_mask(H: np.ndarray, before_hw: tuple[int, int], after_hw: tuple[int, int]) -> np.ndarray:
    """Warped all-ones mask, eroded so interpolated edge pixels count as invalid."""
    ones = np.full(before_hw, 255, np.uint8)
    warped = cv2.warpPerspective(
        ones, H, (after_hw[1], after_hw[0]), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    k = 2 * VALID_ERODE_PX + 1
    # Default erode border treats outside-the-frame as "valid", so only edges
    # against unwarped (zero) pixels shrink.
    warped = cv2.erode(warped, np.ones((k, k), np.uint8))
    return warped > 0


# --------------------------------------------------------------------------- public API


def register(before: np.ndarray, after: np.ndarray, method: str = "sift", work_long: int = 1024) -> RegResult:
    """Estimate H (before pixels -> after pixels) from local features.

    Matching runs at work_long so it costs the same for 12 MP and 1 MP pairs and
    so both images present features at a similar scale. A homography that fails
    the sanity rules falls back to a similarity transform, then to identity
    (ok=False), because a wrong warp manufactures change everywhere.
    """
    _check_image(before, "before")
    _check_image(after, "after")
    if method not in ("sift", "orb"):
        raise CywsError(f"registration method must be 'sift' or 'orb', got {method!r}")
    if int(work_long) < 64:
        raise CywsError(f"work_long must be >= 64, got {work_long}")
    hb, wb = before.shape[:2]
    ha, wa = after.shape[:2]
    cv2.setRNGSeed(RNG_SEED)  # reproducible estimate (cv2's RNG is per thread)

    gb, S_b = _to_work(_gray(before), int(work_long))
    ga, S_a = _to_work(_gray(after), int(work_long))
    det, kind = _detector(method)
    kp_b, des_b = det.detectAndCompute(gb, None)
    kp_a, des_a = det.detectAndCompute(ga, None)
    good = _match(des_b, des_a, kind)

    def lift(Hw: np.ndarray) -> np.ndarray:
        H = np.linalg.inv(S_a) @ Hw @ S_b
        return H / H[2, 2]

    def accept(Hw: np.ndarray, n_in: int, ratio: float, how: str) -> RegResult:
        H = lift(Hw)
        return RegResult(H=H, valid_mask=_valid_mask(H, (hb, wb), (ha, wa)), inliers=n_in,
                         inlier_ratio=ratio, ok=True, method=how, reason=f"{kind}: {len(good)} matches")

    reasons: list[str] = []
    n_in, ratio = 0, 0.0
    if len(good) < 4:
        reasons.append(f"{kind}: only {len(good)} matches")
    else:
        src = np.float32([kp_b[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_a[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        Hw, inl = _find_homography(src, dst)
        n_in = int(inl.sum())
        ratio = n_in / len(good)
        why = "no homography found" if Hw is None else (
            _inlier_reason(n_in, ratio) or _shape_reason(Hw, gb.shape[1], gb.shape[0]))
        if not why:
            return accept(Hw, n_in, ratio, "homography")
        reasons.append(f"homography rejected: {why}")

        M, inl_a = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=REPROJ_THR)
        if M is None:
            reasons.append("affine rejected: no estimate")
        else:
            Aw = np.vstack([M, [0.0, 0.0, 1.0]])
            na = int(inl_a.sum())
            ra = na / len(good)
            why = _inlier_reason(na, ra) or _shape_reason(Aw, gb.shape[1], gb.shape[0])
            if not why:
                return accept(Aw, na, ra, "affine")
            reasons.append(f"affine rejected: {why}")

    # Identity in the canonical sense: the plain resize of before onto after.
    H = _scale_matrix(wa / wb, ha / hb)
    return RegResult(H=H, valid_mask=_valid_mask(H, (hb, wb), (ha, wa)), inliers=n_in,
                     inlier_ratio=ratio, ok=False, method="identity", reason="; ".join(reasons))


def warp_before(before: np.ndarray, reg: RegResult, after: np.ndarray) -> np.ndarray:
    """Warp before into after's frame; uncovered pixels are copied from after.

    Filling from after makes the border read as "no change" instead of a black
    band the model would flag. A before image much larger than after is first
    area-downscaled, since INTER_LINEAR alone aliases fine texture into noise.
    """
    _check_image(before, "before")
    _check_image(after, "after")
    ha, wa = after.shape[:2]
    if reg.valid_mask.shape != (ha, wa):
        raise CywsError(f"valid_mask shape {reg.valid_mask.shape} != after frame {(ha, wa)}")
    if before.ndim != after.ndim:
        raise CywsError("before and after must have the same number of channels")
    H = np.asarray(reg.H, np.float64)
    src = before
    scale = float(np.sqrt(abs(np.linalg.det(H[:2, :2]) / H[2, 2] ** 2)))
    if scale < 0.67:
        hb, wb = before.shape[:2]
        nw, nh = max(1, round(wb * scale)), max(1, round(hb * scale))
        src = cv2.resize(before, (nw, nh), interpolation=cv2.INTER_AREA)
        H = H @ np.linalg.inv(_scale_matrix(nw / wb, nh / hb))
    out = cv2.warpPerspective(src, H, (wa, ha), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    invalid = ~reg.valid_mask
    out[invalid] = after[invalid]
    return out


def photometric_match(src: np.ndarray, ref: np.ndarray, valid_mask: Optional[np.ndarray]) -> np.ndarray:
    """Histogram-match each channel of src to ref, using only the valid region.

    Removes global exposure / white-balance differences so lighting alone does
    not read as change. Statistics come from valid pixels only, because the
    filled border is a copy of ref and would bias the match towards identity.
    """
    _check_image(src, "src")
    _check_image(ref, "ref")
    if src.shape != ref.shape:
        raise CywsError(f"src {src.shape} and ref {ref.shape} must have the same shape")
    if valid_mask is None:
        valid_mask = np.ones(src.shape[:2], bool)
    if valid_mask.shape != src.shape[:2]:
        raise CywsError(f"valid_mask shape {valid_mask.shape} != image {src.shape[:2]}")
    valid_mask = valid_mask.astype(bool)
    if valid_mask.sum() < 64:
        return src.copy()
    s3 = src.reshape(src.shape[0], src.shape[1], -1)
    r3 = ref.reshape(ref.shape[0], ref.shape[1], -1)
    out = np.empty_like(s3)
    levels = np.arange(256, dtype=np.float64)
    for c in range(s3.shape[2]):
        cdf_s = np.cumsum(np.bincount(s3[..., c][valid_mask], minlength=256)).astype(np.float64)
        cdf_r = np.cumsum(np.bincount(r3[..., c][valid_mask], minlength=256)).astype(np.float64)
        cdf_s /= cdf_s[-1]
        cdf_r /= cdf_r[-1]
        lut = np.clip(np.rint(np.interp(cdf_s, cdf_r, levels)), 0, 255).astype(np.uint8)
        out[..., c] = cv2.LUT(s3[..., c], lut)
    return out.reshape(src.shape)


def _abs_diff_u8(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """uint8 |gray(a) - gray(b)|; exact, since both grays are uint8 already."""
    if a.shape != b.shape:
        raise CywsError(f"shape mismatch {a.shape} vs {b.shape}")
    return cv2.absdiff(_gray(a), _gray(b))


def abs_diff_gray(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """float32 |gray(a) - gray(b)|, the change signal used for gating and verification."""
    return _abs_diff_u8(a, b).astype(np.float32)


def median_abs_diff(after: np.ndarray, before_aligned: np.ndarray,
                    valid_mask: Optional[np.ndarray] = None) -> float:
    """Median of abs_diff_gray over valid pixels: verify_detection's global_median_diff.

    Computed from a 256-bin histogram because np.median on a 12 MP float image
    costs ~100 ms on the host and several times that on the board's A65 cores.
    The median is interpolated inside its bin (bin k spans [k - 0.5, k + 0.5)),
    so it is a float. verify_detection compares it with a *mean* residual, and
    for low-noise pairs the integer median (often exactly 1) sits far below the
    noise mean, which would keep perfectly aligned shifted copies; the
    interpolated value tracks the continuous median (mean/median ~1.2 for
    Gaussian noise). Returns 0.0 if nothing is valid.
    """
    d = _abs_diff_u8(after, before_aligned)
    mask = None
    if valid_mask is not None:
        if valid_mask.shape != d.shape:
            raise CywsError(f"valid_mask shape {valid_mask.shape} != image {d.shape}")
        mask = valid_mask.astype(np.uint8)
    hist = cv2.calcHist([d], [0], mask, [256], [0, 256]).ravel().astype(np.float64)
    n = float(hist.sum())
    if n <= 0:
        return 0.0
    cdf = np.cumsum(hist)
    k = int(np.searchsorted(cdf, n / 2.0))
    below = cdf[k] - hist[k]
    lo = max(0.0, k - 0.5)          # |diff| >= 0, so bin 0 only spans [0, 0.5)
    hi = k + 0.5
    return float(lo + (hi - lo) * (n / 2.0 - below) / hist[k])


def diff_energy_map(after: np.ndarray, before_aligned: np.ndarray,
                    valid_mask: Optional[np.ndarray] = None, scale: int = 4) -> np.ndarray:
    """Mean |gray diff| per scale x scale cell, (ceil(H/scale), ceil(W/scale)) float32.

    Same grid as tiling.HeatCanvas so tiles can be gated on either signal.
    Invalid pixels count as zero; partial edge cells average over real pixels.
    Cell sums come from an integral image, which is exact and cheap at 12 MP.
    """
    if scale < 1:
        raise CywsError(f"scale must be >= 1, got {scale}")
    d = _abs_diff_u8(after, before_aligned)
    if valid_mask is not None:
        if valid_mask.shape != d.shape:
            raise CywsError(f"valid_mask shape {valid_mask.shape} != image {d.shape}")
        d = np.where(valid_mask.astype(bool), d, np.uint8(0))
    h, w = d.shape
    ii = cv2.integral(d, sdepth=cv2.CV_64F)
    ys = np.minimum(np.arange(0, h + scale, scale), h)
    xs = np.minimum(np.arange(0, w + scale, scale), w)
    ys = ys[: -(-h // scale) + 1]
    xs = xs[: -(-w // scale) + 1]
    g = ii[np.ix_(ys, xs)]
    sums = g[1:, 1:] - g[:-1, 1:] - g[1:, :-1] + g[:-1, :-1]
    counts = np.outer(np.diff(ys), np.diff(xs))
    return (sums / counts).astype(np.float32)


def _resize_by(img: np.ndarray, s: float) -> tuple[np.ndarray, float, float]:
    """Area-downscale by s; returns (image, actual x factor, actual y factor)."""
    if s >= 1.0:
        return img, 1.0, 1.0
    h, w = img.shape[:2]
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA), nw / w, nh / h


def _zncc(region: np.ndarray, tmpl: np.ndarray) -> np.ndarray:
    res = cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED)
    return np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)


def _zncc_same(a: np.ndarray, b: np.ndarray) -> float:
    """ZNCC of two equal-size patches; matchTemplate would take a slow DFT path here."""
    a0 = np.ascontiguousarray(a, np.float32).ravel()
    b0 = np.ascontiguousarray(b, np.float32).ravel()
    a0 = a0 - np.float32(a0.mean(dtype=np.float64))
    b0 = b0 - np.float32(b0.mean(dtype=np.float64))
    den = float(np.sqrt(float(np.dot(a0, a0)) * float(np.dot(b0, b0))))
    return float(np.dot(a0, b0)) / den if den > 0 else -1.0


def _best_shift(tmpl: np.ndarray, region: np.ndarray) -> tuple[int, int, float]:
    """Full-resolution top-left (ox, oy) of tmpl's best ZNCC match in region.

    matchTemplate over the whole search area costs box area x search area, so
    big boxes are searched coarse to fine: an exhaustive search at a scale where
    the box side is <= VERIFY_MAX_SIDE, then a +-3 px search at each doubling of
    scale, ending at full resolution. Ending at full resolution matters: a
    coarse match mapped back by 1/f is off by up to ~1/f px, and the residual
    measured at that wrong offset would make every large shifted copy look like
    a change. Refinement passes match only a central patch of at most
    2 * VERIFY_MAX_SIDE scaled pixels, which pins down a local shift as well as
    the whole box at a fraction of the cost. The returned ZNCC is the
    full-resolution score of the whole box at (ox, oy).
    """
    bh, bw = tmpl.shape[:2]
    max_ox, max_oy = region.shape[1] - bw, region.shape[0] - bh
    levels = max(0, int(np.ceil(np.log2(max(bw, bh) / float(VERIFY_MAX_SIDE)))))
    ox = oy = None
    zmax = -1.0
    for k in range(levels, -1, -1):
        s = 0.5 ** k
        if ox is None:
            x0, y0, px, py, t_full, sub = 0, 0, 0, 0, tmpl, region
        else:
            side = int(np.ceil(2 * VERIFY_MAX_SIDE / s))
            pw, ph = min(bw, side), min(bh, side)
            px, py = (bw - pw) // 2, (bh - ph) // 2
            t_full = tmpl[py:py + ph, px:px + pw]
            r = int(np.ceil(3.0 / s))
            x0, y0 = max(0, ox - r), max(0, oy - r)
            x1, y1 = min(max_ox, ox + r), min(max_oy, oy + r)
            sub = region[y0 + py:y1 + py + ph, x0 + px:x1 + px + pw]
        t_s, tsx, tsy = _resize_by(t_full, s)
        r_s, rsx, rsy = _resize_by(sub, s)
        if r_s.shape[0] < t_s.shape[0] or r_s.shape[1] < t_s.shape[1]:
            r_s = cv2.resize(sub, (max(r_s.shape[1], t_s.shape[1]), max(r_s.shape[0], t_s.shape[0])),
                             interpolation=cv2.INTER_AREA)
            rsx, rsy = r_s.shape[1] / sub.shape[1], r_s.shape[0] / sub.shape[0]
        _, zmax, _, loc = cv2.minMaxLoc(_zncc(r_s, t_s))
        # Pixel-centre mapping: scaled index i covers full-res (i + 0.5)/f - 0.5,
        # and the template's first scaled pixel sits at 0.5/f_t - 0.5 in it.
        fx = (loc[0] + 0.5) / rsx - 0.5 / tsx
        fy = (loc[1] + 0.5) / rsy - 0.5 / tsy
        ox = int(np.clip(round(x0 + fx), 0, max_ox))
        oy = int(np.clip(round(y0 + fy), 0, max_oy))
    if levels:
        zmax = _zncc_same(region[oy:oy + bh, ox:ox + bw], tmpl)
    return ox, oy, float(np.clip(zmax, -1.0, 1.0))


def verify_detection(det: Detection, after: np.ndarray, before_aligned: np.ndarray, *,
                     zncc_thr: float = 0.85, diff_ratio: float = 1.5,
                     global_median_diff: float, shift_frac: float = 0.03,
                     min_shift: int = 4) -> tuple[bool, dict]:
    """Decide whether a box is a real change or just a locally shifted copy.

    The after patch is searched for in before_aligned over +-shift pixels. The
    box is dropped (keep=False) iff the best ZNCC exceeds zncc_thr *and* the
    mean |diff| at that best shift is below diff_ratio * global_median_diff,
    i.e. the same structure is there, merely displaced (parallax or residual
    misregistration). Using the residual at the best shift, not at zero shift,
    is what lets a shifted-but-unchanged region pass the diff test; the
    zero-shift value is reported as info["diff_raw"]. global_median_diff is
    floored at MIN_GLOBAL_DIFF so near-identical images do not make every box
    unconditionally "changed". Flat after patches (no structure for ZNCC) are
    judged on the diff test alone.
    """
    if after.shape != before_aligned.shape:
        raise CywsError(f"after {after.shape} and before_aligned {before_aligned.shape} differ")
    H, W = after.shape[:2]
    x1 = int(np.clip(np.floor(det.x1), 0, W))
    y1 = int(np.clip(np.floor(det.y1), 0, H))
    x2 = int(np.clip(np.ceil(det.x2), 0, W))
    y2 = int(np.clip(np.ceil(det.y2), 0, H))
    bw, bh = x2 - x1, y2 - y1
    info: dict = {"box_px": [x1, y1, x2, y2]}
    if bw < MIN_VERIFY_SIDE or bh < MIN_VERIFY_SIDE:
        info["reason"] = "small box, kept"
        return True, info

    shift = int(round(max(min_shift, shift_frac * max(bw, bh))))
    sx1, sy1 = max(0, x1 - shift), max(0, y1 - shift)
    sx2, sy2 = min(W, x2 + shift), min(H, y2 + shift)
    tmpl = _gray(after[y1:y2, x1:x2]).astype(np.float32)
    region = _gray(before_aligned[sy1:sy2, sx1:sx2]).astype(np.float32)

    flat = float(tmpl.std()) < FLAT_STD
    if flat:
        zncc_max, ox, oy = 1.0, x1 - sx1, y1 - sy1
    else:
        ox, oy, zncc_max = _best_shift(tmpl, region)

    best = region[oy:oy + bh, ox:ox + bw]
    zero = region[y1 - sy1:y1 - sy1 + bh, x1 - sx1:x1 - sx1 + bw]
    diff_res = float(np.mean(np.abs(tmpl - best)))
    diff_raw = float(np.mean(np.abs(tmpl - zero)))
    diff_thr = float(diff_ratio) * max(float(global_median_diff), MIN_GLOBAL_DIFF)

    drop = zncc_max > zncc_thr and diff_res < diff_thr
    info.update(
        zncc_max=round(zncc_max, 4),
        shift=(ox - (x1 - sx1), oy - (y1 - sy1)),
        search=shift,
        diff_residual=round(diff_res, 3),
        diff_raw=round(diff_raw, 3),
        diff_thr=round(diff_thr, 3),
        flat=flat,
        reason="shifted copy, dropped" if drop else "change confirmed",
    )
    return not drop, info
