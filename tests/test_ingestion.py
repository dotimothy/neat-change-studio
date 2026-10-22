"""ingestion: EXIF handling, mask reconciliation, pair loading, preprocessing."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from common import CywsError, PairItem, Window
from ingestion import (
    apply_exif_orientation,
    crop_pair,
    load_pair,
    preprocess,
    read_image_rgb,
    read_mask,
    resize_to,
)

# Expected display transform of the stored pixels per the EXIF spec, written
# independently of the implementation under test.
EXPECTED = {
    1: lambda a: a,
    2: lambda a: a[:, ::-1],                       # mirror horizontal
    3: lambda a: np.rot90(a, 2),                   # rotate 180
    4: lambda a: a[::-1],                          # mirror vertical
    5: lambda a: np.rot90(a[:, ::-1], 1),          # mirror horizontal + rotate 270 CW
    6: lambda a: np.rot90(a, -1),                  # rotate 90 CW
    7: lambda a: np.rot90(a[:, ::-1], -1),         # mirror horizontal + rotate 90 CW
    8: lambda a: np.rot90(a, 1),                   # rotate 270 CW
}


def quadrant_image(w: int = 160, h: int = 96) -> np.ndarray:
    """Four flat, very different colour blocks: any wrong flip is obvious."""
    img = np.zeros((h, w, 3), np.uint8)
    img[: h // 2, : w // 2] = (230, 20, 20)
    img[: h // 2, w // 2:] = (20, 230, 20)
    img[h // 2:, : w // 2] = (20, 20, 230)
    img[h // 2:, w // 2:] = (230, 230, 30)
    return img


def save(path: Path, arr: np.ndarray, orientation: int | None = None, **kw) -> Path:
    img = Image.fromarray(arr)
    if orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = orientation
        kw["exif"] = exif.tobytes()
    if path.suffix.lower() in (".jpg", ".jpeg"):
        kw.setdefault("quality", 95)
    img.save(path, **kw)
    return path


def win(x0: int, y0: int, w: int, h: int) -> Window:
    return Window(id=0, level=0, x0=x0, y0=y0, w=w, h=h, img_edges=(True, True, True, True))


# ---------------------------------------------------------------- read_image_rgb

@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_orientations_jpeg(tmp_path, orientation):
    raw = quadrant_image()
    path = save(tmp_path / f"o{orientation}.jpg", raw, orientation)

    rgb, meta = read_image_rgb(path)
    expected = EXPECTED[orientation](raw)

    assert rgb.dtype == np.uint8 and rgb.flags.c_contiguous
    assert rgb.shape == expected.shape
    assert np.abs(rgb.astype(int) - expected.astype(int)).mean() < 4.0
    assert meta["exif_orientation"] == orientation
    assert meta["raw_size"] == (160, 96)
    assert meta["size"] == (expected.shape[1], expected.shape[0])


@pytest.mark.parametrize("orientation", range(1, 9))
def test_apply_exif_orientation_matches_spec(orientation):
    raw = np.arange(5 * 7 * 2).reshape(5, 7, 2)
    np.testing.assert_array_equal(apply_exif_orientation(raw, orientation), EXPECTED[orientation](raw))


def test_read_image_modes_to_rgb(tmp_path):
    gray = (np.arange(80 * 64) % 256).astype(np.uint8).reshape(64, 80)
    rgb, meta = read_image_rgb(save(tmp_path / "g.png", gray))
    assert rgb.shape == (64, 80, 3)
    np.testing.assert_array_equal(rgb[..., 0], gray)
    assert meta["exif_orientation"] == 1 and meta["mode"] == "L"

    rgba = np.dstack([quadrant_image(80, 64), np.full((64, 80), 128, np.uint8)])
    rgb, _ = read_image_rgb(save(tmp_path / "a.png", rgba))
    np.testing.assert_array_equal(rgb, quadrant_image(80, 64))

    g16 = np.full((64, 80), 65535, np.uint16)
    rgb, _ = read_image_rgb(save(tmp_path / "g16.png", g16))
    assert rgb.dtype == np.uint8 and int(rgb.min()) == 255


def test_read_image_errors(tmp_path):
    with pytest.raises(CywsError, match="does not exist"):
        read_image_rgb(tmp_path / "missing.jpg")
    bad = tmp_path / "bad.jpg"
    bad.write_text("not an image")
    with pytest.raises(CywsError, match="cannot read"):
        read_image_rgb(bad)
    with pytest.raises(CywsError, match=">= 64"):
        read_image_rgb(save(tmp_path / "small.png", np.zeros((63, 200, 3), np.uint8)))


# ---------------------------------------------------------------------- read_mask

def raw_mask(w: int = 200, h: int = 120) -> np.ndarray:
    """Asymmetric blobs in the raw frame so every orientation is distinguishable."""
    m = np.zeros((h, w), np.uint8)
    m[10:40, 20:70] = 255
    m[70:110, 150:190] = 255
    return m


@pytest.mark.parametrize("orientation", range(1, 9))
def test_raw_size_mask_gets_after_transpose(tmp_path, orientation):
    raw = raw_mask()
    expected = EXPECTED[orientation](raw) > 127
    target_hw = expected.shape
    path = save(tmp_path / "m.png", raw)

    mask = read_mask(path, target_hw, (200, 120), orientation)

    assert mask.dtype == bool and mask.shape == target_hw
    np.testing.assert_array_equal(mask, expected)


def test_canonical_size_mask_used_as_is(tmp_path):
    raw = raw_mask()
    canonical = EXPECTED[6](raw)                 # 120 wide x 200 high
    path = save(tmp_path / "m.png", canonical)
    mask = read_mask(path, (200, 120), (200, 120), 6)
    np.testing.assert_array_equal(mask, canonical > 127)


def test_same_aspect_other_size_is_resized(tmp_path):
    raw = raw_mask()
    half = raw[::2, ::2]                          # 100x60, raw frame, orientation 1
    mask = read_mask(save(tmp_path / "h.png", half), (120, 200), (200, 120), 1)
    assert mask.shape == (120, 200)
    assert (mask == (raw > 127)).mean() > 0.98

    # Half-size mask in the canonical frame of a rotated after image.
    canonical = EXPECTED[6](raw)
    mask = read_mask(save(tmp_path / "c.png", canonical[::2, ::2]), (200, 120), (200, 120), 6)
    assert (mask == (canonical > 127)).mean() > 0.98

    # Half-size mask in the raw frame of a rotated after image: resize, then transpose.
    mask = read_mask(save(tmp_path / "r.png", half), (200, 120), (200, 120), 6)
    assert mask.shape == (200, 120)
    assert (mask == (canonical > 127)).mean() > 0.98


def test_wrong_aspect_mask_raises(tmp_path):
    path = save(tmp_path / "sq.png", np.zeros((150, 150), np.uint8))
    with pytest.raises(CywsError, match="aspect"):
        read_mask(path, (120, 200), (200, 120), 1)


def test_mask_threshold_opening_and_speck_removal(tmp_path):
    m = np.zeros((1000, 1000), np.uint8)
    m[100:300, 100:300] = 200     # kept
    m[500:520, 500:520] = 100     # below threshold
    m[700, 700] = 255             # single pixel: removed by the opening
    m[800:808, 800:808] = 255     # 64 px < 0.01 % of 1e6 px: dropped as a speck
    mask = read_mask(save(tmp_path / "t.png", m), (1000, 1000), (1000, 1000), 1)

    assert mask[100:300, 100:300].all()
    assert mask.sum() == 200 * 200


def test_jpeg_mask_noise_is_cleaned(tmp_path):
    raw = raw_mask()
    mask = read_mask(save(tmp_path / "m.jpg", raw), (120, 200), (200, 120), 1)
    assert (mask == (raw > 127)).mean() > 0.995


# ---------------------------------------------------------------------- load_pair

def make_item(tmp_path: Path, before: np.ndarray, after: np.ndarray, *, mask=None,
              after_orientation=None) -> PairItem:
    b = save(tmp_path / "x_ref.png", before)
    a = save(tmp_path / "x_cap.jpg", after, after_orientation)
    m = save(tmp_path / "x.png", mask) if mask is not None else None
    return PairItem("x", b, a, m)


def test_load_pair_same_aspect_resizes_before(tmp_path):
    item = make_item(tmp_path, quadrant_image(320, 192), quadrant_image(160, 96))
    pair = load_pair(item)
    assert pair.before.shape == pair.after.shape == (96, 160, 3)
    assert pair.meta["before_resized"] and not pair.meta["before_native"]
    assert pair.meta["before"]["size"] == (320, 192)
    assert pair.gt_mask is None


def test_load_pair_within_two_percent_resizes(tmp_path):
    item = make_item(tmp_path, quadrant_image(404, 300), quadrant_image(400, 300))
    pair = load_pair(item)
    assert pair.before.shape == (300, 400, 3)
    assert pair.meta["before_resized"] and not pair.meta["squashed"]


def test_load_pair_aspect_mismatch_rules(tmp_path):
    item = make_item(tmp_path, quadrant_image(300, 300), quadrant_image(400, 300))

    with pytest.raises(CywsError, match="--register"):
        load_pair(item)

    native = load_pair(item, register=True)
    assert native.before.shape == (300, 300, 3)
    assert native.after.shape == (300, 400, 3)
    assert native.meta["before_native"] is True

    squashed = load_pair(item, allow_squash=True)
    assert squashed.before.shape == (300, 400, 3)
    assert squashed.meta["squashed"] and squashed.meta["warnings"]

    both = load_pair(item, register=True, allow_squash=True)
    assert both.meta["before_native"] is True  # registration beats squashing


def test_load_pair_exif_after_defines_frame_and_mask(tmp_path):
    after_raw = quadrant_image(200, 120)
    before = EXPECTED[6](after_raw)              # already upright, 120x200
    item = make_item(tmp_path, before, after_raw, mask=raw_mask(), after_orientation=6)

    pair = load_pair(item)

    assert pair.after.shape == (200, 120, 3)
    assert pair.before.shape == (200, 120, 3) and not pair.meta["before_resized"]
    assert pair.meta["after"]["exif_orientation"] == 6
    assert pair.gt_mask.shape == (200, 120)
    np.testing.assert_array_equal(pair.gt_mask, EXPECTED[6](raw_mask()) > 127)


# ------------------------------------------------------------- preprocess / crops

def test_preprocess_output_contract():
    img = np.random.default_rng(0).integers(0, 256, (300, 500, 3), dtype=np.uint8)
    for w in (win(0, 0, 500, 300), win(10, 20, 256, 256), win(100, 50, 128, 128)):
        for interp in ("area", "bilinear"):
            out = preprocess(img, w, interp)
            assert out.dtype == np.float32
            assert out.shape == (256, 256, 3)
            assert out.flags.c_contiguous
            assert 0.0 <= out.min() and out.max() <= 1.0
            assert not np.shares_memory(out, img)


def test_preprocess_exact_window_is_plain_scaling():
    img = np.random.default_rng(1).integers(0, 256, (300, 400, 3), dtype=np.uint8)
    out = preprocess(img, win(30, 40, 256, 256))
    np.testing.assert_allclose(out, img[40:296, 30:286].astype(np.float32) / 255, atol=1e-7)


def test_area_vs_bilinear():
    # 1-px checkerboard shrunk 3x: bilinear samples single pixels (aliases),
    # area averages 3x3 blocks into near-grey.
    yy, xx = np.mgrid[:768, :768]
    board = (((yy + xx) % 2) * 255).astype(np.uint8)
    img = np.repeat(board[:, :, None], 3, axis=2)
    w = win(0, 0, 768, 768)
    assert preprocess(img, w, "bilinear").std() > 0.4
    assert preprocess(img, w, "area").std() < 0.1

    # Growing: area mode falls back to bilinear exactly.
    small = np.random.default_rng(2).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    np.testing.assert_array_equal(
        preprocess(small, win(0, 0, 128, 128), "area"),
        preprocess(small, win(0, 0, 128, 128), "bilinear"),
    )


def test_preprocess_validation():
    img = np.zeros((100, 100, 3), np.uint8)
    with pytest.raises(CywsError, match="outside"):
        preprocess(img, win(50, 50, 64, 64))
    with pytest.raises(CywsError, match="uint8"):
        preprocess(img.astype(np.float32), win(0, 0, 64, 64))
    with pytest.raises(CywsError, match="interp"):
        preprocess(img, win(0, 0, 64, 64), "cubic")


def test_crop_pair_uses_identical_slices():
    # Encode coordinates in R/G so any slice mismatch shows up; B differs per image.
    yy, xx = np.mgrid[:400, :600]
    coords = np.dstack([xx % 251, yy % 251, np.zeros_like(xx)]).astype(np.uint8)
    before = coords.copy()
    after = coords.copy()
    before[..., 2] = 10
    after[..., 2] = 200
    w = win(137, 61, 300, 280)

    tb, ta = crop_pair(before, after, w)

    np.testing.assert_array_equal(tb[..., :2], ta[..., :2])
    np.testing.assert_array_equal(tb, preprocess(before, w))
    np.testing.assert_array_equal(ta, preprocess(after, w))
    assert np.allclose(tb[..., 2], 10 / 255) and np.allclose(ta[..., 2], 200 / 255)


def test_crop_pair_shape_mismatch_raises():
    with pytest.raises(CywsError, match="identical shapes"):
        crop_pair(np.zeros((100, 120, 3), np.uint8), np.zeros((100, 100, 3), np.uint8),
                  win(0, 0, 64, 64))


def test_resize_to_choice_and_noop():
    img = np.random.default_rng(3).integers(0, 256, (100, 200, 3), dtype=np.uint8)
    assert resize_to(img, 200, 100) is img
    assert resize_to(img, 50, 25).shape == (25, 50, 3)
    assert resize_to(img, 400, 200).shape == (200, 400, 3)
    with pytest.raises(CywsError):
        resize_to(img, 0, 10)
