"""Parallel pair loading: header prediction, equivalence with the serial decode,
GT masks cleaned at working resolution, and dataset prefetch."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import app as app_mod
import backends
import ingestion
from common import AppConfig, CywsError, PairItem
from ingestion import load_pair, read_image_rgb
from test_speed import save, smooth_image


@pytest.mark.parametrize("size, work_long, orientation", [
    ((4208, 3120), 1920, 1), ((4000, 3000), 1920, 6), ((1920, 1080), 1920, 1),
    ((3000, 4000), 1024, 8), ((700, 500), 1920, 1), ((4208, 3120), 0, 1),
])
def test_peek_predicts_the_decode(tmp_path, size, work_long, orientation):
    p = save(tmp_path / "a.jpg", smooth_image(*size), orientation)
    peek = ingestion._peek(p, work_long)
    rgb, meta = read_image_rgb(p, work_long)
    assert peek["raw_size"] == meta["raw_size"]
    assert peek["exif_orientation"] == meta["exif_orientation"] == orientation
    assert peek["long"] == max(rgb.shape[:2])
    assert peek["work_hw"] == rgb.shape[:2]


def test_peek_of_a_bad_file_is_none(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not a jpeg")
    assert ingestion._peek(bad, 1920) is None
    with pytest.raises(CywsError):
        load_pair(PairItem("x", bad, bad), work_long=1920)


@pytest.mark.parametrize("before_size, after_size", [((4208, 3120), (4208, 3120)), ((4000, 3000), (4208, 3156))])
def test_parallel_load_matches_serial(tmp_path, before_size, after_size):
    b = save(tmp_path / "b.jpg", smooth_image(*before_size, seed=1))
    a = save(tmp_path / "a.jpg", smooth_image(*after_size, seed=2))
    pair = load_pair(PairItem("p", b, a), work_long=1920)
    after, _ = read_image_rgb(a, 1920)
    before, _ = read_image_rgb(b, max(after.shape[:2]))
    if before.shape != after.shape:
        before = ingestion.resize_to(before, after.shape[1], after.shape[0])
    assert np.array_equal(pair.after, after)
    assert np.array_equal(pair.before, before)


def test_mask_cleaned_at_working_resolution(tmp_path):
    W, H = 4208, 3120
    gt = np.zeros((H, W), np.uint8)
    cv2.rectangle(gt, (1000, 800), (1600, 1300), 255, -1)    # a real change region
    cv2.circle(gt, (3000, 2000), 300, 255, -1)
    gt[100:102, 100:102] = 255                                 # a speck the opening removes
    img = smooth_image(W, H)
    b, a = save(tmp_path / "b.jpg", img), save(tmp_path / "a.jpg", img)
    m = save(tmp_path / "m.jpg", np.repeat(gt[:, :, None], 3, axis=2))
    pair = load_pair(PairItem("p", b, a, m), work_long=1920)
    h, w = pair.after.shape[:2]
    assert pair.gt_mask.shape == (h, w) and pair.gt_mask.dtype == bool
    # the old route: clean at native, then nearest-resize
    native = ingestion.read_mask(m, (H, W), (W, H), 1)
    old = cv2.resize(native.view(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    assert (pair.gt_mask != old).mean() < 0.002
    assert not pair.gt_mask[:5, :5].any()
    assert pair.meta["mask"]["read_ms"] > 0


def test_mask_follows_after_orientation(tmp_path):
    W, H = 4000, 3000                                          # raw (stored) after size
    gt = np.zeros((H, W), np.uint8)
    gt[:600, :800] = 255                                       # raw top-left
    img = smooth_image(W, H)
    b, a = save(tmp_path / "b.jpg", img, 6), save(tmp_path / "a.jpg", img, 6)
    m = save(tmp_path / "m.jpg", np.repeat(gt[:, :, None], 3, axis=2))
    pair = load_pair(PairItem("p", b, a, m), work_long=1920)
    h, w = pair.after.shape[:2]
    assert h > w                                               # portrait after the transpose
    native = ingestion.read_mask(m, (W, H), (W, H), 6)        # target is (H, W) = (4000, 3000)
    old = cv2.resize(native.view(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    assert (pair.gt_mask != old).mean() < 0.002


def test_dataset_prefetch_keeps_order_and_survives_a_bad_pair(tmp_path):
    img = smooth_image(640, 480)
    items = []
    for i in range(4):
        b, a = save(tmp_path / f"{i}_b.jpg", img), save(tmp_path / f"{i}_a.jpg", img)
        items.append(PairItem(f"p{i}", b, a))
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"nope")
    items.insert(2, PairItem("bad", bad, bad))
    be = backends.FakeBackend()
    try:
        app = app_mod.CywsApp(be, AppConfig(levels=(0,), work_long=0))
        logs: list[str] = []
        summary = app_mod.run_dataset(app, items, tmp_path / "out", log=logs.append)
    finally:
        be.close()
    lines = [l for l in logs if l.startswith("[")]
    assert [l.split("] ")[1].split(":")[0] for l in lines] == ["p0", "p1", "bad", "p2", "p3"]
    assert "FAILED" in lines[2]
    assert summary["n_ok"] == 4 and summary["n_failed"] == 1
    assert any(l.startswith("wall time") for l in logs)


def test_dataset_background_write_failure_marks_the_pair_failed(tmp_path, monkeypatch):
    img = smooth_image(640, 480)
    items = []
    for i in range(3):
        b, a = save(tmp_path / f"{i}_b.jpg", img), save(tmp_path / f"{i}_a.jpg", img)
        items.append(PairItem(f"p{i}", b, a))
    real = app_mod.post_process.save_outputs

    def flaky(result, after, out_dir, **kw):
        if result.pair_id == "p1":
            raise CywsError("disk full")
        return real(result, after, out_dir, **kw)

    monkeypatch.setattr(app_mod.post_process, "save_outputs", flaky)
    be = backends.FakeBackend()
    try:
        app = app_mod.CywsApp(be, AppConfig(levels=(0,), work_long=0))
        logs: list[str] = []
        summary = app_mod.run_dataset(app, items, tmp_path / "out", log=logs.append)
    finally:
        be.close()
    assert summary["n_ok"] == 2 and summary["n_failed"] == 1
    assert any("p1: FAILED: disk full" in l for l in logs)
    assert (tmp_path / "out" / "p2_overlay.jpg").exists() and (tmp_path / "out" / "p0.json").exists()
    rows = list(__import__("csv").DictReader(open(tmp_path / "out" / "summary.csv")))
    assert [r["status"] for r in rows if r["pair_id"] != "mean"][:3] == ["ok", "failed", "ok"]
    for pid in ("p0", "p2"):
        assert all((tmp_path / "out" / f"{pid}{suffix}").exists() for suffix in ("_overlay.jpg", "_mask.png", ".json"))
