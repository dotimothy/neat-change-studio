"""Tests for app/post_process.py on synthetic outputs, boxes and masks."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from common import CH, AppConfig, CywsError, Detection, RegResult, Result, Window
import post_process as pp

ALL_EDGES = (True, True, True, True)


# ----------------------------------------------------------------- helpers

def gaussian_output(peaks: list[dict], branch: str = "after", sigma: float = 2.0) -> np.ndarray:
    """Canonical (10,256,256) output with Gaussian heat peaks and wh/off at each peak."""
    out = np.zeros((10, 256, 256), np.float32)
    yy, xx = np.mgrid[0:256, 0:256]
    sfx = "a" if branch == "after" else "b"
    for p in peaks:
        g = p["score"] * np.exp(-((xx - p["x"]) ** 2 + (yy - p["y"]) ** 2) / (2 * sigma**2))
        out[CH[f"heat_{sfx}"]] = np.maximum(out[CH[f"heat_{sfx}"]], g)
        out[CH[f"wh_{sfx}_w"], p["y"], p["x"]] = p["w"]
        out[CH[f"wh_{sfx}_h"], p["y"], p["x"]] = p["h"]
        out[CH[f"off_{sfx}_x"], p["y"], p["x"]] = p["dx"]
        out[CH[f"off_{sfx}_y"], p["y"], p["x"]] = p["dy"]
    return out


def det(x1, y1, x2, y2, score=0.5, branch="after", window_id=0, level=0, label=None) -> Detection:
    return Detection(float(x1), float(y1), float(x2), float(y2), score, branch, window_id, level, label)


def no_suppress_cfg(**kw) -> AppConfig:
    """Config with greedy suppression disabled, to isolate the huge/split rules."""
    return AppConfig(nms_iou=1.01, containment=1.01, **kw)


W0 = Window(0, 0, 0, 0, 1024, 1024, ALL_EDGES)
W1 = Window(1, 1, 0, 0, 512, 512, (True, True, False, False))
W2 = Window(2, 1, 384, 0, 512, 512, (False, True, False, False))
WINDOWS = [W0, W1, W2]


# ----------------------------------------------------------------- decode

def test_decode_recovers_boxes_exactly():
    peaks = [
        dict(x=50, y=60, score=0.9, w=20.0, h=30.0, dx=0.3, dy=0.7),
        dict(x=150, y=200, score=0.7, w=40.0, h=10.0, dx=0.0, dy=0.25),
        dict(x=220, y=30, score=0.5, w=8.0, h=12.0, dx=0.9, dy=0.1),
    ]
    out = gaussian_output(peaks)
    boxes = pp.decode_centernet(out, "after", thr=0.3, topk=100)
    assert boxes.dtype == np.float32 and boxes.shape == (3, 5)
    for row, p in zip(boxes, peaks):  # sorted by score desc == peaks order
        cx, cy = p["x"] + p["dx"], p["y"] + p["dy"]
        expected = [cx - p["w"] / 2, cy - p["h"] / 2, cx + p["w"] / 2, cy + p["h"] / 2, p["score"]]
        np.testing.assert_allclose(row, expected, atol=1e-4)
    # The other branch is empty.
    assert pp.decode_centernet(out, "before").shape == (0, 5)


def test_decode_threshold_topk_and_local_max():
    out = np.zeros((10, 256, 256), np.float32)
    h = out[CH["heat_b"]]
    h[100, 100] = 0.8
    h[100, 101] = 0.6      # neighbour of a stronger peak: not a local max
    h[10, 10] = 0.95
    h[200, 50] = 0.4
    h[30, 200] = 0.25      # below threshold
    boxes = pp.decode_centernet(out, "before", thr=0.3, topk=100)
    np.testing.assert_allclose(boxes[:, 4], [0.95, 0.8, 0.4], atol=1e-6)
    top2 = pp.decode_centernet(out, "before", thr=0.3, topk=2)
    np.testing.assert_allclose(top2[:, 4], [0.95, 0.8], atol=1e-6)
    # Zero wh gives a point box at the peak.
    np.testing.assert_allclose(top2[0, :4], [10, 10, 10, 10])


def test_decode_validates_inputs():
    with pytest.raises(CywsError):
        pp.decode_centernet(np.zeros((10, 128, 128), np.float32), "after")
    with pytest.raises(CywsError):
        pp.decode_centernet(np.zeros((10, 256, 256), np.float32), "sideways")
    with pytest.raises(CywsError):
        pp.decode_centernet(np.zeros((10, 256, 256), np.float32), "after", topk=0)


# ----------------------------------------------------------------- filter_window

def test_filter_window_truncation_interior_vs_border():
    cfg = AppConfig(edge_margin=2.0, min_box_side=6.0)
    d = np.array([
        [1.0, 50, 40, 90, 0.9],      # touches left = image border -> kept
        [50, 1.0, 90, 40, 0.8],      # touches top = image border -> kept
        [200, 50, 255, 90, 0.7],     # touches right = interior -> dropped
        [50, 200, 90, 254.5, 0.6],   # touches bottom = interior -> dropped
        [100, 100, 253.9, 150, 0.5], # 2.1 px from interior right -> kept
    ], np.float32)
    kept = pp.filter_window(d, W1, "after", cfg)
    assert [round(k.score, 2) for k in kept] == [0.9, 0.8, 0.5]
    # At level 0 the truncation rule does not apply, even on interior sides.
    w0_interior = Window(5, 0, 0, 0, 256, 256, (False, False, False, False))
    assert len(pp.filter_window(d, w0_interior, "after", cfg)) == 5


def test_filter_window_min_side_clip_and_mapping():
    cfg = AppConfig(min_box_side=6.0)
    win = Window(3, 0, 100, 50, 512, 512, ALL_EDGES)
    d = np.array([
        [64, 64, 128, 128, 0.9],
        [10, 10, 15, 100, 0.8],      # 5 wide -> dropped
        [10, 10, 16, 100, 0.7],      # 6 wide -> kept
        [-10, 20, 50, 80, 0.6],      # clipped to x1 = 0
        [-30, 20, 3, 80, 0.5],       # 3 wide after clipping -> dropped
    ], np.float32)
    kept = pp.filter_window(d, win, "before", cfg)
    assert [round(k.score, 2) for k in kept] == [0.9, 0.7, 0.6]
    k0 = kept[0]
    assert (k0.x1, k0.y1, k0.x2, k0.y2) == (100 + 128, 50 + 128, 100 + 256, 50 + 256)
    assert (k0.branch, k0.window_id, k0.level) == ("before", 3, 0)
    assert kept[2].x1 == 100.0 and kept[2].x2 == 100 + 100.0
    assert pp.filter_window(np.zeros((0, 5), np.float32), win, "after", cfg) == []


def test_drop_invalid():
    vm = np.zeros((100, 100), bool)
    vm[:, :50] = True
    dets = [det(60, 10, 90, 40, 0.9), det(20, 10, 70, 40, 0.8), det(10, 10, 40, 40, 0.7)]
    kept = pp.drop_invalid(dets, vm)
    assert [d.score for d in kept] == [0.8, 0.7]   # second box is 60% valid
    assert pp.drop_invalid(dets, None) == dets


# ----------------------------------------------------------------- box matrices

def test_iou_and_containment_matrices():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[5, 0, 15, 10], [2, 2, 4, 4], [20, 20, 30, 30]], float)
    np.testing.assert_allclose(pp.iou_matrix(a, b), [[50 / 150, 4 / 100, 0]])
    # containment[i, j] = inter / area(b[j])
    np.testing.assert_allclose(pp.containment_matrix(a, b), [[0.5, 1.0, 0.0]])
    np.testing.assert_allclose(pp.containment_matrix(b, a), [[0.5], [0.04], [0.0]])
    assert pp.iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)


# ----------------------------------------------------------------- merge

def test_merge_removes_cross_window_duplicates_per_branch():
    dets = [
        det(400, 100, 480, 180, 0.9, window_id=1, level=1),
        det(402, 101, 482, 181, 0.8, window_id=2, level=1),
        det(400, 100, 480, 180, 0.7, branch="before", window_id=1, level=1),
    ]
    out = pp.merge(dets, WINDOWS, AppConfig())
    assert [(d.branch, d.score) for d in out] == [("after", 0.9), ("before", 0.7)]


def test_merge_containment_is_directional():
    big, small = (100, 100, 300, 300), (150, 150, 200, 200)
    out = pp.merge([det(*big, 0.9), det(*small, 0.5)], [W0], AppConfig())
    assert [d.score for d in out] == [0.9]
    # A lower-scoring box that contains a higher-scoring one is kept.
    out = pp.merge([det(*big, 0.5), det(*small, 0.9)], [W0], AppConfig())
    assert sorted(d.score for d in out) == [0.5, 0.9]


def test_merge_huge_box_rule():
    huge = det(0, 0, 500, 500, 0.9, window_id=1, level=1)          # 95% of window 1
    out = pp.merge([huge], WINDOWS, no_suppress_cfg())
    assert out == []
    support = det(0, 0, 480, 480, 0.6, window_id=0, level=0)       # IoU 0.92
    out = pp.merge([huge, support], WINDOWS, no_suppress_cfg())
    assert sorted(d.score for d in out) == [0.6, 0.9]
    weak = det(600, 600, 1000, 1000, 0.6, window_id=0, level=0)    # IoU 0
    out = pp.merge([huge, weak], WINDOWS, no_suppress_cfg())
    assert [d.score for d in out] == [0.6]
    # Without any level-0 window there is nothing to support it: rule skipped.
    assert len(pp.merge([huge], [W1, W2], no_suppress_cfg())) == 1


def test_merge_split_rule():
    coarse = det(0, 0, 600, 600, 0.6, window_id=0, level=0)
    a = det(10, 10, 110, 110, 0.8, window_id=1, level=1)
    b = det(400, 400, 500, 500, 0.7, window_id=2, level=1)
    out = pp.merge([coarse, a, b], WINDOWS, no_suppress_cfg())
    assert [d.score for d in out] == [0.8, 0.7]
    # A single child does not split the coarse box.
    out = pp.merge([coarse, a], WINDOWS, no_suppress_cfg())
    assert sorted(d.score for d in out) == [0.6, 0.8]
    # Children that tile the coarse box (union cover 1.0) keep it.
    left = det(0, 0, 300, 400, 0.8, window_id=1, level=1)
    right = det(300, 0, 600, 400, 0.7, window_id=2, level=1)
    coarse2 = det(0, 0, 600, 400, 0.6, window_id=0, level=0)
    out = pp.merge([coarse2, left, right], WINDOWS, no_suppress_cfg())
    assert sorted(d.score for d in out) == [0.6, 0.7, 0.8]


def test_merge_wbf_option():
    dets = [det(100, 100, 200, 200, 0.9), det(110, 100, 210, 200, 0.6)]
    out = pp.merge(dets, [W0], AppConfig(merge="wbf"))
    assert len(out) == 1
    d = out[0]
    assert d.score == pytest.approx(0.9)
    assert (d.x1, d.x2) == (pytest.approx(104.0), pytest.approx(204.0))
    assert (d.y1, d.y2) == (pytest.approx(100.0), pytest.approx(200.0))
    with pytest.raises(CywsError):
        pp.merge(dets, [W0], AppConfig(merge="soft"))  # type: ignore[arg-type]


def test_merge_cap():
    dets = [det(i * 50, 0, i * 50 + 40, 40, score=0.1 + i * 0.05) for i in range(10)]
    out = pp.merge(dets, [W0], AppConfig(max_boxes=3))
    assert [round(d.score, 2) for d in out] == [0.55, 0.5, 0.45]


# Level 0 of a 1920x1080 image: two 1080-px squares sharing the strip x 840..1080.
SQ0 = Window(0, 0, 0, 0, 1080, 1080, (True, True, False, True))
SQ1 = Window(1, 0, 840, 0, 1080, 1080, (False, True, True, True))


def test_merge_rejoins_object_cut_by_level0_seam():
    left = det(696, 295, 1080, 705, 0.8, window_id=0)       # reaches SQ0's interior right side
    right = det(840, 295, 1304, 705, 0.7, window_id=1)      # reaches SQ1's interior left side
    out = pp.merge([left, right], [SQ0, SQ1], AppConfig())
    assert len(out) == 1
    d = out[0]
    assert (d.x1, d.y1, d.x2, d.y2) == (696, 295, 1304, 705)
    assert d.score == 0.8 and d.window_id == 0


def test_merge_seam_rule_leaves_unrelated_boxes():
    # Both reach the seam but lie at different heights: two objects, not one.
    a = det(900, 100, 1080, 300, 0.8, window_id=0)
    b = det(840, 600, 1000, 800, 0.7, window_id=1)
    assert len(pp.merge([a, b], [SQ0, SQ1], AppConfig())) == 2
    # A box well inside its square is never touched.
    c = det(200, 200, 400, 400, 0.6, window_id=0)
    assert len(pp.merge([c, b], [SQ0, SQ1], AppConfig())) == 2
    # Same boxes from a single full-image window: no seam, no rejoining.
    full = Window(0, 0, 0, 0, 1920, 1080, ALL_EDGES)
    out = pp.merge([det(696, 295, 1080, 705, 0.8), det(840, 295, 1304, 705, 0.7)], [full], AppConfig())
    assert len(out) == 2


def test_merge_seam_rule_drops_fragment_seen_whole_elsewhere():
    frag = det(980, 300, 1080, 500, 0.9, window_id=0)       # cut by SQ0's right side
    whole = det(900, 280, 1200, 520, 0.6, window_id=1)      # SQ1 sees the object whole
    out = pp.merge([frag, whole], [SQ0, SQ1], AppConfig())
    assert [(d.x1, d.x2, d.score) for d in out] == [(900, 1200, 0.6)]


def test_merge_seam_rule_chains_stacked_squares():
    # 64x3000 image: 47 stacked 64-px squares; one object spans ten of them.
    import tiling
    wins = [w for w in tiling.plan_windows(3000, 64, levels=(0,))]
    dets = []
    for w in wins:
        y1, y2 = max(1200, w.y0), min(1800, w.y1)
        if y2 > y1:
            dets.append(det(10, y1, 50, y2, 0.5 + 0.001 * w.id, window_id=w.id))
    assert len(dets) >= 9
    out = pp.merge(dets, wins, AppConfig())
    assert [(d.x1, d.y1, d.x2, d.y2) for d in out] == [(10, 1200, 50, 1800)]


# ----------------------------------------------------------------- fuse_branches

def test_fuse_branches_labels():
    after = [det(10, 10, 50, 50, 0.6, window_id=1), det(200, 200, 250, 250, 0.4)]
    before = [det(12, 10, 52, 50, 0.8, branch="before"), det(300, 300, 350, 350, 0.5, branch="before")]
    out = pp.fuse_branches(after, before, iou=0.5)
    by_label = {d.label: d for d in out}
    assert set(by_label) == {"changed", "appeared", "disappeared"} and len(out) == 3
    ch = by_label["changed"]
    assert (ch.x1, ch.x2, ch.score, ch.branch, ch.window_id) == (10, 50, 0.8, "after", 1)
    assert by_label["appeared"].x1 == 200
    assert by_label["disappeared"].x1 == 300 and by_label["disappeared"].branch == "before"
    assert [d.score for d in out] == sorted((d.score for d in out), reverse=True)
    assert after[0].label is None  # inputs are not mutated
    assert pp.fuse_branches([], []) == []


# ----------------------------------------------------------------- make_mask

def test_make_mask_box_mode():
    m = pp.make_mask([det(2.5, 3, 6, 8)], (10, 10))
    expected = np.zeros((10, 10), bool)
    expected[3:8, 2:6] = True
    np.testing.assert_array_equal(m, expected)
    with pytest.raises(CywsError):
        pp.make_mask([], (10, 10), mode="blob")


def test_make_mask_diff_mode():
    before = np.full((100, 100, 3), 100, np.uint8)
    after = before.copy()
    after[40:60, 40:60] = 200
    box = det(30, 30, 70, 70)
    m = pp.make_mask([box], (100, 100), "diff", after=after, before_aligned=before)
    square = np.zeros((100, 100), bool)
    square[40:60, 40:60] = True
    iou = (m & square).sum() / (m | square).sum()
    assert iou > 0.8
    assert not m[:30].any() and not m[70:].any()
    # No difference inside the box -> falls back to the filled box.
    m2 = pp.make_mask([box], (100, 100), "diff", after=before, before_aligned=before)
    assert m2.sum() == 40 * 40
    # Without before_aligned "diff" degrades to "box".
    m3 = pp.make_mask([box], (100, 100), "diff", after=after)
    assert m3.sum() == 40 * 40


# ----------------------------------------------------------------- metrics

def test_metrics_pixel_and_components():
    gt = np.zeros((10, 10), bool)
    gt[0:4, 0:4] = True          # component 1, 16 px
    gt[6:8, 6:8] = True          # component 2, 4 px
    pred = np.zeros((10, 10), bool)
    pred[0:4, 2:6] = True        # 8 on GT, 8 off
    dets = [det(0, 0, 4, 4), det(8, 8, 10, 10)]
    m = pp.metrics(pred, gt, dets)
    assert (m["tp"], m["fp"], m["fn"]) == (8, 8, 12)
    assert m["iou"] == pytest.approx(8 / 28)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.4)
    assert m["f1"] == pytest.approx(16 / 36)
    assert (m["gt_components"], m["gt_components_hit"]) == (2, 1)
    assert m["component_recall"] == pytest.approx(0.5)
    assert m["box_precision"] == pytest.approx(0.5)
    # A box over component 2: 4/9 of it on GT (>= 30%) and covers it fully.
    m = pp.metrics(pred, gt, dets + [det(5, 5, 8, 8)])
    assert m["gt_components_hit"] == 2 and m["box_precision"] == pytest.approx(2 / 3)


def test_metrics_hit_by_bbox_iou_and_empty_cases():
    gt = np.zeros((10, 10), bool)
    gt[0:5, 0:5] = True
    gt[1:4, 1:4] = False         # hollow ring, bbox (0,0,5,5)
    inner = det(1, 1, 4, 4)      # covers no ring pixel, bbox IoU 9/25
    m = pp.metrics(np.zeros_like(gt), gt, [inner])
    assert m["gt_components_hit"] == 1
    assert m["box_precision"] == 0.0
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0
    tiny = det(2, 2, 3, 3)       # bbox IoU 1/25
    assert pp.metrics(np.zeros_like(gt), gt, [tiny])["gt_components_hit"] == 0
    empty = np.zeros((10, 10), bool)
    m = pp.metrics(empty, empty, [])
    assert all(m[k] == 1.0 for k in ("iou", "precision", "recall", "f1", "component_recall", "box_precision"))
    with pytest.raises(CywsError):
        pp.metrics(np.zeros((5, 5), bool), empty, [])


# ----------------------------------------------------------------- overlay / save

def make_result(h=120, w=160) -> Result:
    heat = np.zeros(((h + 3) // 4, (w + 3) // 4), np.float32)
    heat[5:10, 5:10] = 0.9
    dets = [
        det(10, 10, 60, 50, 0.9, label="appeared"),
        det(80, 20, 150, 90, 0.6, branch="before", label="disappeared"),
        det(20, 70, 70, 110, 0.5, label="changed", window_id=1, level=1),
    ]
    mask = pp.make_mask(dets, (h, w))
    reg = RegResult(np.eye(3), np.ones((h, w), bool), 120, 0.8, True, "homography")
    return Result(
        pair_id="pair/01", detections=dets, mask=mask, heat=heat,
        windows=[Window(0, 0, 0, 0, w, h, ALL_EDGES)], registration=reg,
        timings_ms={"total": 12.5, "infer": np.float32(4.25)},
        meta={"path": Path("/x/y.jpg"), "size": (w, h), "n": np.int64(3), "bad": float("nan"),
              "arr": np.arange(3)},
    )


def test_overlay_returns_uint8_rgb_same_shape():
    rng = np.random.default_rng(0)
    after = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    res = make_result()
    vis = pp.overlay(after, res)
    assert vis.dtype == np.uint8 and vis.shape == after.shape
    assert not np.array_equal(vis, after)
    # Empty result still renders (legend only).
    empty = replace(res, detections=[], mask=np.zeros((120, 160), bool), heat=np.zeros((30, 40), np.float32))
    assert pp.overlay(after, empty).shape == after.shape
    with pytest.raises(CywsError):
        pp.overlay(after[..., 0], res)


def test_save_outputs_writes_valid_files(tmp_path):
    after = np.full((120, 160, 3), 128, np.uint8)
    res = make_result()
    paths = pp.save_outputs(res, after, tmp_path / "out", save_heat=True)
    assert set(paths) == {"overlay", "mask", "json", "heat"}
    for p in paths.values():
        assert Path(p).is_file() and Path(p).parent == tmp_path / "out"
    assert Path(paths["json"]).name == "pair_01.json"

    overlay_img = cv2.imread(paths["overlay"])
    assert overlay_img.shape == (120, 160, 3)
    mask_img = cv2.imread(paths["mask"], cv2.IMREAD_GRAYSCALE)
    assert set(np.unique(mask_img)) <= {0, 255}
    np.testing.assert_array_equal(mask_img > 127, res.mask)
    assert cv2.imread(paths["heat"], cv2.IMREAD_GRAYSCALE).shape == res.heat.shape

    doc = json.loads(Path(paths["json"]).read_text())
    assert doc["pair_id"] == "pair/01"
    assert len(doc["detections"]) == 3 and doc["detections"][0]["label"] == "appeared"
    assert doc["windows"][0]["w"] == 160
    assert doc["registration"]["ok"] is True and doc["registration"]["valid_frac"] == 1.0
    assert doc["timings_ms"]["infer"] == 4.25
    assert doc["meta"] == {"path": "/x/y.jpg", "size": [160, 120], "n": 3, "bad": None, "arr": [0, 1, 2]}

    # Without save_heat no heat file is written.
    paths2 = pp.save_outputs(res, after, tmp_path / "out2")
    assert "heat" not in paths2
