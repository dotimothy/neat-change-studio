"""Tests for app/registration.py on synthetic scenes with known geometry."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import registration as reg
from common import CywsError, Detection, RegResult


def textured_scene(h: int, w: int, seed: int = 0, n_shapes: int = 400) -> np.ndarray:
    """Non-repetitive RGB texture: random shapes and lines over smooth noise."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (h // 16 + 1, w // 16 + 1, 3), dtype=np.uint8)
    img = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    for _ in range(n_shapes):
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        kind = rng.integers(0, 3)
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(4, max(6, min(h, w) // 20)))
        if kind == 0:
            cv2.circle(img, (x, y), r, color, -1, cv2.LINE_AA)
        elif kind == 1:
            cv2.rectangle(img, (x, y), (x + r, y + int(rng.integers(4, 2 * r))), color, -1)
        else:
            cv2.line(img, (x, y), (int(rng.integers(0, w)), int(rng.integers(0, h))), color, 2, cv2.LINE_AA)
    return img


def similarity(angle_deg: float, scale: float, cx: float, cy: float) -> np.ndarray:
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)
    return np.vstack([M, [0, 0, 1]])


def make_pair(after_hw=(900, 1200), before_scale=1.0, angle=4.0, zoom=1.05,
              persp=(4e-5, -3e-5), seed=1):
    """after is a crop of a big scene; before re-images the scene with a known homography.

    Returns (before, after, H_true) with H_true mapping before pixels -> after pixels.
    """
    ha, wa = after_hw
    margin = 300
    scene = textured_scene(ha + 2 * margin, wa + 2 * margin, seed)
    T_after = np.array([[1, 0, -margin], [0, 1, -margin], [0, 0, 1]], float)  # scene -> after
    after = cv2.warpPerspective(scene, T_after, (wa, ha))
    # scene -> before: perspective + rotation/zoom about the after centre, then optional rescale.
    cx, cy = margin + wa / 2, margin + ha / 2
    P = np.array([[1, 0, 0], [0, 1, 0], [persp[0], persp[1], 1]], float)
    C = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], float)
    M_b = similarity(angle, zoom, 0, 0) @ np.linalg.inv(C) @ P @ C
    M_b = T_after @ M_b                                           # frame the same region
    M_b = reg._scale_matrix(before_scale, before_scale) @ M_b
    hb, wb = round(ha * before_scale), round(wa * before_scale)
    before = cv2.warpPerspective(scene, M_b, (wb, hb), flags=cv2.INTER_AREA if before_scale < 1 else cv2.INTER_LINEAR)
    H_true = T_after @ np.linalg.inv(M_b)
    return before, after, H_true / H_true[2, 2]


def corner_error(H_est: np.ndarray, H_true: np.ndarray, before_hw) -> float:
    h, w = before_hw
    pts = np.float64([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
    a = cv2.perspectiveTransform(pts, H_est).reshape(-1, 2)
    b = cv2.perspectiveTransform(pts, H_true).reshape(-1, 2)
    return float(np.linalg.norm(a - b, axis=1).mean())


@pytest.mark.parametrize("method", ["sift", "orb"])
def test_recovers_known_homography(method):
    before, after, H_true = make_pair()
    r = reg.register(before, after, method=method, work_long=1024)
    assert r.ok, r.reason
    assert r.method == "homography"
    assert r.inliers >= reg.MIN_INLIERS and r.inlier_ratio >= reg.MIN_INLIER_RATIO
    assert corner_error(r.H, H_true, before.shape[:2]) < 2.0
    assert r.valid_mask.shape == after.shape[:2] and r.valid_mask.dtype == bool
    assert 0.5 < r.valid_mask.mean() < 1.0  # rotation leaves an uncovered border


@pytest.mark.parametrize("method", ["sift", "orb"])
def test_full_res_lift_with_downscaled_work(method):
    """work_long below the image size exercises the S_after^-1 @ H @ S_before lift."""
    before, after, H_true = make_pair(after_hw=(1200, 1600), angle=-3.0, zoom=0.97, seed=3)
    r = reg.register(before, after, method=method, work_long=800)
    assert r.ok, r.reason
    assert corner_error(r.H, H_true, before.shape[:2]) < 2.0


@pytest.mark.parametrize("before_scale", [2.0, 0.6])
def test_different_size_before(before_scale):
    before, after, H_true = make_pair(before_scale=before_scale, seed=5)
    assert before.shape[:2] != after.shape[:2]
    r = reg.register(before, after, method="sift")
    assert r.ok, r.reason
    assert corner_error(r.H, H_true, before.shape[:2]) < 2.0
    warped = reg.warp_before(before, r, after)
    assert warped.shape == after.shape
    d = reg.abs_diff_gray(warped, after)[r.valid_mask]
    assert np.median(d) < 8.0


def test_textureless_falls_back_to_identity():
    rng = np.random.default_rng(0)
    after = np.full((400, 600, 3), 128, np.uint8)
    before = np.clip(np.full((800, 1200, 3), 120.0) + rng.normal(0, 0.3, (800, 1200, 3)), 0, 255).astype(np.uint8)
    r = reg.register(before, after)
    assert not r.ok
    assert r.method == "identity"
    assert r.reason
    np.testing.assert_allclose(r.H, reg._scale_matrix(0.5, 0.5), atol=1e-12)
    assert r.valid_mask.all()
    # Same-size identity is the true identity.
    r2 = reg.register(after, after.copy())
    assert not r2.ok and np.allclose(r2.H, np.eye(3))


def test_rejects_bad_input():
    img = np.zeros((100, 100, 3), np.uint8)
    with pytest.raises(CywsError):
        reg.register(img.astype(np.float32), img)
    with pytest.raises(CywsError):
        reg.register(img, img, method="akaze")
    with pytest.raises(CywsError):
        reg.register(np.zeros((8, 100, 3), np.uint8), img)


def test_warp_before_fills_invalid_from_after():
    after = textured_scene(300, 400, seed=7)
    before = textured_scene(300, 400, seed=8)
    H = np.array([[1, 0, 60], [0, 1, -30], [0, 0, 1]], float)  # before shifted right/up
    valid = reg._valid_mask(H, before.shape[:2], after.shape[:2])
    r = RegResult(H=H, valid_mask=valid, inliers=100, inlier_ratio=1.0, ok=True, method="homography")
    out = reg.warp_before(before, r, after)
    assert not valid[:, :60].any() and not valid[-30:, :].any()
    assert valid[:-32, 62:].all()
    np.testing.assert_array_equal(out[~valid], after[~valid])
    # Interior valid pixels are the translated before (integer shift -> exact).
    np.testing.assert_array_equal(out[50:200, 100:300], before[80:230, 40:240])


def test_photometric_match_reduces_colour_error():
    ref = textured_scene(300, 400, seed=9)
    f = ref.astype(np.float32) / 255.0
    src = np.clip(255 * (0.8 * f ** 1.4 + np.array([0.05, -0.02, 0.08])), 0, 255).astype(np.uint8)
    valid = np.ones(ref.shape[:2], bool)
    valid[:, :80] = False
    src[:, :80] = 255  # garbage outside the valid region must not skew the match
    before_err = np.abs(src.astype(float) - ref)[valid].mean()
    out = reg.photometric_match(src, ref, valid)
    after_err = np.abs(out.astype(float) - ref)[valid].mean()
    assert out.dtype == np.uint8 and out.shape == src.shape
    assert after_err < 0.3 * before_err
    assert after_err < 4.0


def test_diff_energy_map_shape_and_validity():
    a = np.zeros((101, 203, 3), np.uint8)
    b = a.copy()
    b[:40, :40] = 100
    valid = np.ones(a.shape[:2], bool)
    valid[:, 100:] = False
    b[:, 150:] = 200
    m = reg.diff_energy_map(a, b, valid, scale=4)
    assert m.shape == (26, 51) and m.dtype == np.float32
    assert m[:10, :10].min() > 90 and m[:, 26:].max() == 0.0


def _noisy(img: np.ndarray, rng: np.random.Generator, sigma: float = 3.0) -> np.ndarray:
    return np.clip(img + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def test_verify_detection_drops_shift_keeps_real_change():
    rng = np.random.default_rng(11)
    scene = textured_scene(600, 800, seed=12)
    before_aligned = _noisy(scene, rng)
    after_clean = scene.copy()
    # Region A: unchanged content displaced by 3 px (parallax / residual misregistration).
    after_clean[100:220, 100:260] = scene[97:217, 103:263]
    # Region B: a genuinely new object pasted in.
    after_clean[350:470, 450:610] = textured_scene(120, 160, seed=99, n_shapes=60)
    after = _noisy(after_clean, rng)
    gmd = float(np.median(reg.abs_diff_gray(after, before_aligned)))

    shifted = Detection(100, 100, 260, 220, 0.9, "after")
    keep, info = reg.verify_detection(shifted, after, before_aligned, global_median_diff=gmd)
    assert not keep, info
    assert info["zncc_max"] > 0.85 and info["shift"] == (3, -3)
    assert info["diff_raw"] > info["diff_thr"]  # a zero-shift diff alone would have kept it

    real = Detection(450, 350, 610, 470, 0.9, "after")
    keep, info = reg.verify_detection(real, after, before_aligned, global_median_diff=gmd)
    assert keep, info

    unchanged = Detection(500, 50, 700, 200, 0.5, "after")
    keep, _ = reg.verify_detection(unchanged, after, before_aligned, global_median_diff=gmd)
    assert not keep

    tiny = Detection(10, 10, 15, 40, 0.5, "after")
    keep, info = reg.verify_detection(tiny, after, before_aligned, global_median_diff=gmd)
    assert keep and "small" in info["reason"]


def test_verify_detection_large_box_uses_reduced_search():
    rng = np.random.default_rng(13)
    scene = textured_scene(900, 1200, seed=14)
    before_aligned = _noisy(scene, rng)
    after = _noisy(np.roll(scene, (5, -4), axis=(0, 1)), rng)
    gmd = float(np.median(reg.abs_diff_gray(after, before_aligned)))
    box = Detection(200, 150, 900, 750, 0.8, "after")
    keep, info = reg.verify_detection(box, after, before_aligned, global_median_diff=gmd)
    assert not keep, info
    assert abs(info["shift"][0] - 4) <= 1 and abs(info["shift"][1] + 5) <= 1


def _smooth_texture(h: int, w: int, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    g = cv2.GaussianBlur(rng.uniform(0, 255, (h, w)).astype(np.float32), (0, 0), sigma)
    g = cv2.normalize(g, None, 20, 235, cv2.NORM_MINMAX)
    return np.dstack([g] * 3)


@pytest.mark.parametrize("side", [300, 600, 1000])
@pytest.mark.parametrize("sigma", [1.0, 3.0, 6.0])
def test_verify_detection_large_shifted_copy_is_dropped(side, sigma):
    # Large boxes are searched coarse to fine; the shift must come out exact at
    # full resolution, or the residual is measured at a misaligned offset.
    img = np.clip(np.rint(_smooth_texture(side + 200, side + 200, sigma, seed=side)), 0, 255).astype(np.uint8)
    before = img.copy()
    after = img.copy()
    after[100:100 + side, 100:100 + side] = img[100:100 + side, 97:97 + side]
    gmd = reg.median_abs_diff(after, before)
    box = Detection(100, 100, 100 + side, 100 + side, 0.9, "after")
    keep, info = reg.verify_detection(box, after, before, global_median_diff=gmd)
    assert not keep, info
    assert info["shift"] == (-3, 0) and info["diff_residual"] == 0.0


def test_verify_detection_large_box_keeps_real_change():
    img = np.clip(np.rint(_smooth_texture(1000, 1000, 3.0, seed=5)), 0, 255).astype(np.uint8)
    after = img.copy()
    after[300:600, 300:600] = 255 - after[300:600, 300:600]
    gmd = reg.median_abs_diff(after, img)
    keep, info = reg.verify_detection(Detection(250, 250, 650, 650, 0.9, "after"), after, img,
                                      global_median_diff=gmd)
    assert keep, info


@pytest.mark.parametrize("noise", [1.0, 2.0, 3.0, 4.0, 5.0, 8.0])
def test_verify_detection_low_noise_shifted_copy_is_dropped(noise):
    # The box residual is a mean; the global statistic must not be an integer
    # median that sits below the noise mean (it was exactly 1.0 at sigma 3).
    rng = np.random.default_rng(int(noise * 10))
    img = _smooth_texture(800, 1000, 4.0, seed=21)
    after = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    after[200:400, 300:500] = after[200:400, 297:497].copy()
    before = np.clip(np.rint(img + rng.normal(0, noise, img.shape)), 0, 255).astype(np.uint8)
    gmd = reg.median_abs_diff(after, before)
    keep, info = reg.verify_detection(Detection(300, 200, 500, 400, 0.9, "after"), after, before,
                                      global_median_diff=gmd)
    assert not keep, info
    assert info["shift"] == (-3, 0)


def test_median_abs_diff_is_interpolated():
    a = np.zeros((100, 100, 3), np.uint8)
    b = a.copy()
    b[:, :30] = 1          # 30% ones, 70% zeros -> median inside bin 0 = [0, 0.5)
    assert 0.0 < reg.median_abs_diff(a, b) < 0.5
    b[:, :60] = 2          # 60% twos -> median in bin 2, interpolated to 2 - 0.5 + 0.1/0.6
    assert reg.median_abs_diff(a, b) == pytest.approx(1.5 + 0.1 / 0.6, abs=1e-6)
