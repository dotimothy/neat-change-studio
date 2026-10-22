"""Tests for app/tiling.py: window plan coverage, mapping, heat canvas, gating."""

from __future__ import annotations

import math

import numpy as np
import pytest

import tiling
from common import CywsError, Window

SIZES = [
    (4032, 3024), (3024, 4032), (1920, 1080), (1080, 1920), (300, 5000), (5000, 300),
    (200, 180), (64, 64), (256, 256), (257, 1000), (500, 500), (720, 1280), (1000, 1501),
    (999, 1500), (2000, 2000), (333, 777), (4000, 6000), (100, 2000), (256, 4096),
]


def _coverage(H: int, W: int, wins: list[Window]) -> np.ndarray:
    cnt = np.zeros((H, W), dtype=np.uint16)
    for w in wins:
        cnt[w.slices()] += 1
    return cnt


def _win(x0: int, y0: int, w: int, h: int, H: int, W: int, level: int = 1) -> Window:
    return Window(0, level, x0, y0, w, h, (x0 == 0, y0 == 0, x0 + w == W, y0 + h == H))


# ----------------------------------------------------------------- axis_positions
@pytest.mark.parametrize("L,s,ov", [(4032, 1512, 0.25), (3024, 1512, 0.25), (1000, 256, 0.0),
                                    (1000, 256, 0.5), (257, 256, 0.25), (5000, 300, 0.9)])
def test_axis_positions_cover_and_flush(L, s, ov):
    pos = tiling.axis_positions(L, s, ov)
    n_expected = math.ceil((L - s) / (s * (1 - ov))) + 1
    assert len(pos) == n_expected
    assert pos[0] == 0 and pos[-1] + s == L
    assert all(b > a for a, b in zip(pos, pos[1:]))
    gaps = [b - a for a, b in zip(pos, pos[1:])]
    assert max(gaps) <= s  # contiguous coverage
    if ov > 0:
        assert max(gaps) <= s * (1 - ov) + 1  # requested overlap kept (to rounding)


def test_axis_positions_single_and_errors():
    assert tiling.axis_positions(300, 300, 0.25) == [0]
    assert tiling.axis_positions(4032, 1512, 0.25) == [0, 840, 1680, 2520]
    with pytest.raises(CywsError):
        tiling.axis_positions(100, 200, 0.25)
    with pytest.raises(CywsError):
        tiling.axis_positions(100, 50, 1.0)
    with pytest.raises(CywsError):
        tiling.axis_positions(100, 0, 0.25)


# ----------------------------------------------------------------- auto_tile_size
def test_auto_tile_size():
    assert tiling.auto_tile_size(4032, 3024) == 1512
    assert tiling.auto_tile_size(300, 5000) == 256          # clamped up
    assert tiling.auto_tile_size(200, 180) == 180           # short side < 256 -> short side
    assert tiling.auto_tile_size(4032, 3024, 100) == 400
    assert tiling.auto_tile_size(4032, 3024, 10) == 256
    assert tiling.auto_tile_size(4032, 3024, 5000) == 3024  # clamped to short side
    with pytest.raises(CywsError):
        tiling.auto_tile_size(4032, 3024, 0)


# ----------------------------------------------------------------- plan_windows
@pytest.mark.parametrize("H,W", SIZES)
@pytest.mark.parametrize("levels", [(0,), (0, 1), (0, 1, 2)])
def test_plan_covers_every_pixel_per_level(H, W, levels):
    wins = tiling.plan_windows(H, W, levels=levels)
    assert [w.id for w in wins] == list(range(len(wins)))
    assert [w.level for w in wins] == sorted(w.level for w in wins)  # level 0 first
    for lv in sorted({w.level for w in wins}):
        lw = [w for w in wins if w.level == lv]
        for w in lw:
            assert 0 <= w.x0 and 0 <= w.y0 and w.x1 <= W and w.y1 <= H and w.w > 0 and w.h > 0
            assert w.img_edges == (w.x0 == 0, w.y0 == 0, w.x1 == W, w.y1 == H)
        assert (_coverage(H, W, lw) >= 1).all(), f"level {lv} leaves holes in {W}x{H}"
        # last window is flush with the right and bottom edges
        assert max(w.x1 for w in lw) == W and max(w.y1 for w in lw) == H
        if lv >= 1:
            assert all(w.w == w.h for w in lw)


def test_plan_12mp_example():
    wins = tiling.plan_windows(4032, 3024)
    assert len(wins) == 13
    assert wins[0] == Window(0, 0, 0, 0, 3024, 4032, (True, True, True, True))
    l1 = wins[1:]
    assert all(w.level == 1 and w.w == 1512 for w in l1)
    assert sorted({w.x0 for w in l1}) == [0, 756, 1512]
    assert sorted({w.y0 for w in l1}) == [0, 840, 1680, 2520]
    # row-major within a level
    assert [(w.y0, w.x0) for w in l1] == sorted((w.y0, w.x0) for w in l1)
    # rotated image gives the transposed plan
    rot = tiling.plan_windows(3024, 4032)
    assert len(rot) == 13 and sorted({w.x0 for w in rot[1:]}) == [0, 840, 1680, 2520]


def test_plan_extreme_aspect_level0_squares():
    wins = tiling.plan_windows(300, 5000)
    assert all(w.level == 0 for w in wins)  # s1=256 >= 0.8*300 -> level 1 skipped
    assert len(wins) == math.ceil(5000 / 300)
    assert all(w.w == 300 and w.h == 300 and w.y0 == 0 for w in wins)
    assert wins[0].x0 == 0 and wins[-1].x1 == 5000
    assert wins[0].img_edges == (True, True, False, True)
    assert wins[5].img_edges == (False, True, False, True)
    tall = tiling.plan_windows(5000, 300, levels=(0,))
    assert all(w.x0 == 0 and w.w == 300 for w in tall) and tall[-1].y1 == 5000


def test_plan_aspect_threshold():
    assert len(tiling.plan_windows(1000, 1500, levels=(0,))) == 1   # exactly 1.5 -> squash
    assert len(tiling.plan_windows(1000, 1501, levels=(0,))) == 2   # above -> 2 squares
    assert len(tiling.plan_windows(1000, 2000, levels=(0,))) == 2   # exact integer ratio


def test_plan_small_image_level0_only():
    wins = tiling.plan_windows(200, 180, levels=(0, 1, 2))
    assert wins == [Window(0, 0, 0, 0, 180, 200, (True, True, True, True))]
    # even if level 0 was not requested, a small image still gets processed
    assert tiling.plan_windows(200, 180, levels=(1,)) == wins


@pytest.mark.parametrize("H,W", [(63, 1000), (1000, 63), (10, 10)])
def test_plan_too_small_raises(H, W):
    with pytest.raises(CywsError):
        tiling.plan_windows(H, W)


def test_plan_level_rules():
    # level 1 skipped when the tile is near the full short side
    assert {w.level for w in tiling.plan_windows(300, 300)} == {0}            # s1=256 >= 240
    assert {w.level for w in tiling.plan_windows(1000, 1000, tile=900)} == {0}
    # explicit tile is clamped to [256, short]
    l1 = [w for w in tiling.plan_windows(2000, 2000, tile=100) if w.level == 1]
    assert l1 and all(w.w == 256 for w in l1)
    # level 2 only on request and only if s1//2 >= 384
    assert 2 not in {w.level for w in tiling.plan_windows(4032, 3024)}
    l2 = [w for w in tiling.plan_windows(4032, 3024, levels=(0, 1, 2)) if w.level == 2]
    assert l2 and all(w.w == 756 for w in l2)
    assert 2 not in {w.level for w in tiling.plan_windows(1920, 1080, levels=(0, 1, 2))}  # 270 < 384
    # hint drives the auto tile
    l1h = [w for w in tiling.plan_windows(4032, 3024, median_component_long_side=100) if w.level == 1]
    assert all(w.w == 400 for w in l1h)
    # empty plan falls back to level 0
    assert [w.level for w in tiling.plan_windows(300, 300, levels=(1,))] == [0]


@pytest.mark.parametrize("kw", [dict(levels=()), dict(levels=(3,)), dict(overlap=1.0),
                                dict(overlap=-0.1), dict(tile=0), dict(tile="big"),
                                dict(tile=True), dict(max_aspect_squash=0.5)])
def test_plan_bad_args(kw):
    with pytest.raises(CywsError):
        tiling.plan_windows(1000, 1000, **kw)


# ----------------------------------------------------------------- mapping
@pytest.mark.parametrize("H,W", [(4032, 3024), (300, 5000), (200, 180), (1920, 1080)])
def test_mapping_round_trip(H, W):
    rng = np.random.default_rng(0)
    for win in tiling.plan_windows(H, W, levels=(0, 1, 2)):
        b256 = np.sort(rng.uniform(0, 256, (50, 4)).reshape(50, 2, 2), axis=1).reshape(50, 4)
        b256 = np.concatenate([b256[:, [0, 2, 1, 3]], rng.uniform(0, 1, (50, 1))], axis=1)
        b256 = b256.astype(np.float32)
        can = tiling.window_to_canonical(b256, win)
        assert can.shape == b256.shape and can.dtype == np.float64
        assert (can[:, [0, 2]] >= win.x0 - 1e-6).all() and (can[:, [0, 2]] <= win.x1 + 1e-6).all()
        assert (can[:, [1, 3]] >= win.y0 - 1e-6).all() and (can[:, [1, 3]] <= win.y1 + 1e-6).all()
        np.testing.assert_array_equal(can[:, 4], b256[:, 4])  # extra columns copied
        back = tiling.canonical_to_window(can, win)
        assert np.abs(back - b256).max() < 0.5
        # round trip canonical -> window -> canonical too, in pixels
        assert np.abs(tiling.window_to_canonical(back, win) - can).max() < 0.5


def test_mapping_exact_values_and_shapes():
    win = Window(3, 1, 100, 200, 512, 1024, (False, False, False, False))
    out = tiling.window_to_canonical(np.array([[0, 0, 256, 256], [128, 64, 129, 65]]), win)
    np.testing.assert_allclose(out, [[100, 200, 612, 1224], [356, 456, 358, 460]])
    one = tiling.window_to_canonical(np.array([0.0, 0, 256, 256]), win)
    assert one.shape == (4,)
    assert tiling.window_to_canonical(np.zeros((0, 5)), win).shape == (0, 5)
    src = np.array([[1.0, 2, 3, 4]])
    tiling.window_to_canonical(src, win)
    np.testing.assert_array_equal(src, [[1, 2, 3, 4]])  # input not mutated
    with pytest.raises(CywsError):
        tiling.window_to_canonical(np.zeros((3, 3)), win)


def test_valid_fraction():
    mask = np.zeros((100, 200), dtype=bool)
    mask[:, :100] = True
    assert tiling.valid_fraction(_win(50, 0, 100, 100, 100, 200), mask) == pytest.approx(0.5)
    assert tiling.valid_fraction(_win(0, 0, 100, 100, 100, 200), mask) == 1.0
    assert tiling.valid_fraction(_win(100, 0, 100, 100, 100, 200), mask) == 0.0
    with pytest.raises(CywsError):
        tiling.valid_fraction(_win(150, 0, 100, 100, 100, 250), mask)


# ----------------------------------------------------------------- heat canvas
def test_heat_canvas_shape_and_max_semantics():
    H, W = 1001, 1499
    c = tiling.HeatCanvas(H, W)
    assert c.array.shape == (math.ceil(H / 4), math.ceil(W / 4)) and c.array.dtype == np.float32
    full = _win(0, 0, W, H, H, W, level=0)
    c.add(full, np.full((256, 256), 0.2, np.float32))
    np.testing.assert_allclose(c.array, 0.2, atol=1e-6)
    tile = _win(400, 300, 400, 400, H, W)
    c.add(tile, np.full((256, 256), 0.7, np.float32))
    rows, cols = c.footprint(tile)
    assert rows == slice(75, 175) and cols == slice(100, 200)
    np.testing.assert_allclose(c.array[rows, cols], 0.7, atol=1e-6)
    outside = np.ones_like(c.array, dtype=bool)
    outside[rows, cols] = False
    np.testing.assert_allclose(c.array[outside], 0.2, atol=1e-6)
    # a lower heat never overwrites a higher one
    c.add(full, np.full((256, 256), 0.5, np.float32))
    np.testing.assert_allclose(c.array[rows, cols], 0.7, atol=1e-6)
    np.testing.assert_allclose(c.array[outside], 0.5, atol=1e-6)
    assert c.max_in(tile) == pytest.approx(0.7)
    assert c.max_in(_win(0, 0, 200, 200, H, W)) == pytest.approx(0.5)


def test_heat_canvas_keeps_peak_when_shrinking():
    # a 200x180 image: the 256 heat is shrunk to 50x45 cells; a 1-px peak must survive
    c = tiling.HeatCanvas(200, 180)
    heat = np.zeros((256, 256), np.float32)
    heat[200, 37] = 0.9
    c.add(_win(0, 0, 180, 200, 200, 180, level=0), heat)
    assert c.array.max() == pytest.approx(0.9)
    r, q = np.unravel_index(np.argmax(c.array), c.array.shape)
    assert abs(r - 200 * 50 / 256) <= 1 and abs(q - 37 * 45 / 256) <= 1


def test_heat_canvas_peak_location_when_growing():
    H, W = 4032, 3024
    c = tiling.HeatCanvas(H, W)
    heat = np.zeros((256, 256), np.float32)
    heat[64, 192] = 1.0
    win = _win(1512, 840, 1512, 1512, H, W)
    c.add(win, heat[None])  # (1, 256, 256) accepted
    r, q = np.unravel_index(np.argmax(c.array), c.array.shape)
    exp_r = (840 + 64.5 * 1512 / 256) / 4
    exp_q = (1512 + 192.5 * 1512 / 256) / 4
    assert abs(r - exp_r) <= 2 and abs(q - exp_q) <= 2


def test_heat_canvas_errors():
    c = tiling.HeatCanvas(100, 100)
    with pytest.raises(CywsError):
        c.add(_win(50, 50, 100, 100, 150, 150), np.zeros((256, 256)))
    with pytest.raises(CywsError):
        c.add(_win(0, 0, 100, 100, 100, 100), np.zeros((2, 256, 256)))
    with pytest.raises(CywsError):
        tiling.HeatCanvas(0, 10)


# ----------------------------------------------------------------- gating
def _gating_setup():
    H, W = 2000, 2000
    wins = [w for w in tiling.plan_windows(H, W) if w.level == 1]
    assert len(wins) == 9  # s1=1000, 3x3 grid
    return H, W, wins, tiling.HeatCanvas(H, W)


def test_gate_by_heat():
    H, W, wins, c = _gating_setup()
    assert tiling.gate_windows(wins, c) == []
    heat = np.zeros((256, 256), np.float32)
    heat[10:14, 10:14] = 0.5  # top-left corner of the image
    c.add(_win(0, 0, W, H, H, W, level=0), heat)
    kept = tiling.gate_windows(wins, c, gate_heat=0.1)
    assert kept == [wins[0]]
    assert tiling.gate_windows(wins, c, gate_heat=0.6) == []
    assert tiling.gate_windows(wins, c, gate_heat=0.0) == wins


def test_gate_by_diff_energy():
    H, W, wins, c = _gating_setup()
    energy = np.zeros_like(c.array)
    assert tiling.gate_windows(wins, c, energy) == []  # flat map adds nothing
    energy[-20:, -20:] = 50.0  # bottom-right corner
    kept = tiling.gate_windows(wins, c, energy, gate_heat=0.1, diff_percentile=75.0)
    assert kept == [wins[-1]]
    # union with heat gating, order preserved
    heat = np.zeros((256, 256), np.float32)
    heat[10:14, 10:14] = 0.5
    c.add(_win(0, 0, W, H, H, W, level=0), heat)
    assert tiling.gate_windows(wins, c, energy) == [wins[0], wins[-1]]
    assert tiling.gate_windows([], c, energy) == []


def test_gate_errors():
    H, W, wins, c = _gating_setup()
    with pytest.raises(CywsError):
        tiling.gate_windows(wins, c, np.zeros((10, 10), np.float32))
    with pytest.raises(CywsError):
        tiling.gate_windows(wins, c, np.zeros_like(c.array), diff_percentile=101)
