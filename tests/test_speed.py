"""Speed track: reduced-resolution decoding, threaded crops, honest timings.

- read_image_rgb / load_pair with work_long: JPEG draft decoding vs PNG
  power-of-two downscale, EXIF after draft, the before image and the GT mask in
  the working frame, meta["scale"] / meta["orig_size"].
- box_orig in the JSON output.
- threaded crop_pair keeps FIFO order, tags and tensor contents; gated mode
  still streams level 0 first.
- timing keys add up; backend latency is compute time (sync) or push -> output
  (async).
"""

from __future__ import annotations

import csv
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

import backends
import ingestion
import post_process
from common import AppConfig, CywsError, Detection, Pair, PairItem, Result, Window
from ingestion import crop_pair, load_pair, read_image_rgb, reduction_for

ALL_EDGES = (True, True, True, True)


def smooth_image(w: int, h: int, seed: int = 0) -> np.ndarray:
    """Low-frequency colour noise: survives JPEG and downscaling almost unchanged."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (max(2, h // 64), max(2, w // 64), 3), dtype=np.uint8)
    return np.ascontiguousarray(cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC))


def quadrants(w: int, h: int) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    img[: h // 2, : w // 2] = (230, 20, 20)
    img[: h // 2, w // 2:] = (20, 230, 20)
    img[h // 2:, : w // 2] = (20, 20, 230)
    img[h // 2:, w // 2:] = (230, 230, 30)
    return img


def save(path: Path, arr: np.ndarray, orientation: int | None = None) -> Path:
    kw = {}
    if orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = orientation
        kw["exif"] = exif.tobytes()
    if path.suffix.lower() in (".jpg", ".jpeg"):
        kw["quality"] = 95
    Image.fromarray(arr).save(path, **kw)
    return path


# ------------------------------------------------------------------ decoding sizes

@pytest.mark.parametrize(
    "long_side, work_long, cap, expected",
    [(4208, 1920, 8, 2), (4000, 1920, 8, 2), (1920, 1920, 8, 1), (3839, 1920, 8, 1), (3840, 1920, 8, 2),
     (16000, 1920, 8, 8), (16000, 1920, 0, 8), (40000, 1920, 8, 8), (40000, 1920, 0, 16), (4208, 0, 8, 1)],
)
def test_reduction_for(long_side, work_long, cap, expected):
    assert reduction_for(long_side, work_long, max_factor=cap) == expected


@pytest.mark.parametrize(
    "size, work_long, expected, draft",
    [
        ((1000, 750), 256, (500, 375), 2),     # 1000 // 4 = 250 < 256 -> 1/2
        ((1001, 751), 256, (501, 376), 2),     # odd sizes: the decoder rounds up
        ((1000, 750), 200, (250, 188), 4),
        ((1000, 750), 120, (125, 94), 8),
        ((1000, 750), 600, (1000, 750), 1),    # 1000 // 2 = 500 < 600 -> native
        ((1000, 750), 0, (1000, 750), 1),      # 0 = native
    ],
)
def test_jpeg_draft_decoding_sizes(tmp_path, size, work_long, expected, draft):
    w, h = size
    path = save(tmp_path / "a.jpg", smooth_image(w, h))
    rgb, meta = read_image_rgb(path, work_long)
    assert (rgb.shape[1], rgb.shape[0]) == expected and rgb.dtype == np.uint8 and rgb.flags.c_contiguous
    assert meta["draft"] == draft and meta["reduction"] == draft
    assert meta["size"] == expected and meta["full_size"] == (w, h) and meta["raw_size"] == (w, h)
    assert meta["scale"] == pytest.approx((w / expected[0], h / expected[1]))
    if work_long:
        assert max(expected) >= min(work_long, max(size))
    # Draft decoding matches a full decode + area resize closely (smooth content).
    full, _ = read_image_rgb(path)
    ref = cv2.resize(full, expected, interpolation=cv2.INTER_AREA)
    assert np.abs(rgb.astype(int) - ref.astype(int)).mean() < 3.0


def test_jpeg_draft_is_actually_used(tmp_path, monkeypatch):
    """The JPEG path asks PIL for draft decoding; the PNG path never does."""
    from PIL import JpegImagePlugin

    calls = []
    orig_jpeg = JpegImagePlugin.JpegImageFile.draft
    orig_base = Image.Image.draft

    def spy(orig):
        def draft(self, mode, size):
            calls.append((self.format, mode, size))
            return orig(self, mode, size)
        return draft

    monkeypatch.setattr(JpegImagePlugin.JpegImageFile, "draft", spy(orig_jpeg))
    monkeypatch.setattr(Image.Image, "draft", spy(orig_base))
    img = smooth_image(1000, 750)
    read_image_rgb(save(tmp_path / "a.jpg", img), 256)
    read_image_rgb(save(tmp_path / "a.png", img), 256)
    assert calls == [("JPEG", "RGB", (500, 375))]


@pytest.mark.parametrize(
    "size, work_long, expected",
    [
        ((1000, 750), 256, (500, 375)),   # 1000 >= 2 * 256 -> 1/2
        ((1000, 750), 120, (125, 94)),    # 1/8, rounded up like the JPEG decoder
        ((1000, 750), 500, (500, 375)),   # exactly 2 * work_long
        ((1000, 750), 501, (1000, 750)),  # < 2 * work_long: no downscale
        ((1000, 750), 0, (1000, 750)),
    ],
)
def test_png_power_of_two_downscale(tmp_path, size, work_long, expected):
    w, h = size
    img = smooth_image(w, h)
    rgb, meta = read_image_rgb(save(tmp_path / "a.png", img), work_long)
    assert (rgb.shape[1], rgb.shape[0]) == expected
    assert meta["draft"] == 1 and meta["reduction"] == round(w / expected[0])
    if expected != size:
        np.testing.assert_array_equal(rgb, cv2.resize(img, expected, interpolation=cv2.INTER_AREA))
    else:
        np.testing.assert_array_equal(rgb, img)


def test_jpeg_beyond_draft_limit_gets_extra_downscale(tmp_path):
    # 4400 // 8 = 550 >= 2 * 256: draft gives 1/8, then a 1/2 area resize -> 1/16.
    rgb, meta = read_image_rgb(save(tmp_path / "big.jpg", smooth_image(4400, 2200)), 256)
    assert (rgb.shape[1], rgb.shape[0]) == (275, 138)
    assert meta["draft"] == 8 and meta["reduction"] == 16
    assert meta["scale"] == pytest.approx((4400 / 275, 2200 / 138))


def test_too_small_natively_is_rejected(tmp_path):
    with pytest.raises(CywsError, match="must be >= 64"):
        read_image_rgb(save(tmp_path / "thin.png", smooth_image(2048, 50)), 256)
    with pytest.raises(CywsError):
        read_image_rgb(save(tmp_path / "ok.png", smooth_image(300, 200)), -1)


# ------------------------------------------------------------------ elongated images (short-side floor)

@pytest.mark.parametrize(
    "long_side, short_side, work_long, cap, expected",
    [
        (4208, 3120, 1920, 8, 2),     # 4:3 - the short side never binds
        (4000, 3000, 1920, 8, 2),
        (3840, 2160, 1920, 8, 2),     # 16:9 4K: 1080 >= 960
        (7680, 1080, 1920, 8, 1),     # panorama: 1/2 would leave 540 < 960
        (16000, 500, 1920, 8, 1),     # strip: stays native
        (16000, 6, 1920, 8, 1),       # sliver: no reduction, so draft never sees a 0 size
        (16000, 4000, 1920, 8, 4),    # 4:1: the short side stops it at 1/4 (1000), not 1/8 (500)
        (1000, 750, 120, 8, 8),       # short floor is max(64, work_long // 2)
        (2048, 100, 256, 0, 1),
    ],
)
def test_reduction_for_respects_short_side(long_side, short_side, work_long, cap, expected):
    s = reduction_for(long_side, work_long, max_factor=cap, short_side=short_side)
    assert s == expected
    assert ingestion.short_floor_for(work_long) == max(64, work_long // 2)
    if s > 1:
        assert short_side // s >= ingestion.short_floor_for(work_long) and long_side // s >= work_long


def test_aspect_up_to_2_is_unchanged_by_short_floor():
    for long_side in range(1024, 9000, 37):
        for aspect in (1.0, 4 / 3, 16 / 9, 2.0):
            short = int(long_side / aspect)
            assert reduction_for(long_side, 1920, 8, short_side=short) == reduction_for(long_side, 1920, 8)


def test_jpeg_sliver_fails_as_cyws_error_not_zero_division(tmp_path):
    # Regression: draft("RGB", (2000, 0)) raised ZeroDivisionError, which escaped run_dataset.
    path = save(tmp_path / "strip.jpg", smooth_image(16000, 6))
    with pytest.raises(CywsError, match="must be >= 64"):
        read_image_rgb(path, 1920)


def test_open_image_maps_zero_division_to_cyws_error(tmp_path, monkeypatch):
    path = save(tmp_path / "a.jpg", smooth_image(800, 600))

    def boom(self, *a, **k):
        raise ZeroDivisionError("integer division or modulo by zero")

    monkeypatch.setattr(Image.Image, "load", boom)
    with pytest.raises(CywsError, match="cannot read"):
        read_image_rgb(path)


def test_wide_jpeg_keeps_native_short_side(tmp_path):
    # Regression: 16000x500 drafted to 2000x63 and was rejected although valid natively.
    rgb, meta = read_image_rgb(save(tmp_path / "wide.jpg", smooth_image(16000, 500)), 1920)
    assert (rgb.shape[1], rgb.shape[0]) == (16000, 500) and meta["reduction"] == 1


def test_panorama_keeps_level1_tiles(tmp_path):
    # Regression: 7680x1080 decoded to 1920x270 at work_long=1920, which skipped all of level 1.
    from tiling import plan_windows

    img = smooth_image(7680, 1080)
    item = PairItem("pano", save(tmp_path / "b.jpg", img), save(tmp_path / "a.jpg", img))
    pair = load_pair(item, work_long=1920)
    assert pair.after.shape[:2] == (1080, 7680) and pair.before.shape == pair.after.shape
    native = plan_windows(1080, 7680, levels=(0, 1))
    planned = plan_windows(*pair.after.shape[:2], levels=(0, 1))
    assert len(planned) == len(native) and any(w.level == 1 for w in planned)


@pytest.mark.parametrize("orientation", [1, 3, 6, 8])
def test_exif_transpose_after_draft(tmp_path, orientation):
    raw = quadrants(1000, 600)
    path = save(tmp_path / f"o{orientation}.jpg", raw, orientation)
    rgb, meta = read_image_rgb(path, 256)
    expected_full = {1: raw, 3: np.rot90(raw, 2), 6: np.rot90(raw, -1), 8: np.rot90(raw, 1)}[orientation]
    fh, fw = expected_full.shape[:2]
    assert meta["draft"] == 2 and meta["raw_size"] == (1000, 600) and meta["full_size"] == (fw, fh)
    assert (rgb.shape[1], rgb.shape[0]) == (fw // 2, fh // 2) == meta["size"]
    expected = cv2.resize(np.ascontiguousarray(expected_full), (fw // 2, fh // 2), interpolation=cv2.INTER_AREA)
    assert np.abs(rgb.astype(int) - expected.astype(int)).mean() < 4.0
    assert meta["scale"] == pytest.approx((2.0, 2.0))


# ------------------------------------------------------------------ load_pair

def test_load_pair_working_frame_before_and_mask(tmp_path):
    after = smooth_image(1000, 750, seed=1)
    before = smooth_image(2000, 1500, seed=2)          # same aspect, twice the resolution
    gt = np.zeros((750, 1000), np.uint8)
    gt[300:420, 500:700] = 255
    item = PairItem(
        "p",
        before_path=save(tmp_path / "b.png", before),
        after_path=save(tmp_path / "a.jpg", after),
        mask_path=save(tmp_path / "m.png", gt),
    )
    pair = load_pair(item, work_long=256)
    assert pair.after.shape == (375, 500, 3)
    assert pair.before.shape == pair.after.shape       # before resized into the working frame
    assert pair.meta["before"]["reduction"] == 4       # long side >= after's working long side (500)
    assert pair.meta["scale"] == pytest.approx((2.0, 2.0))
    assert pair.meta["orig_size"] == (1000, 750) and pair.meta["work_size"] == (500, 375)
    assert pair.meta["work_long"] == 256
    assert pair.gt_mask.shape == (375, 500) and pair.gt_mask.dtype == bool
    native = ingestion.read_mask(item.mask_path, (750, 1000), (1000, 750), 1)
    expected = cv2.resize(native.astype(np.uint8), (500, 375), interpolation=cv2.INTER_NEAREST).astype(bool)
    np.testing.assert_array_equal(pair.gt_mask, expected)
    assert pair.gt_mask[150:210, 250:350].all() and pair.gt_mask.sum() == 60 * 100
    assert "read_ms" in pair.meta["mask"]

    native_pair = load_pair(item)                      # default work_long=0: unchanged behaviour
    assert native_pair.after.shape == (750, 1000, 3) and native_pair.meta["scale"] == (1.0, 1.0)
    assert native_pair.gt_mask.sum() == 120 * 200


def test_load_pair_mask_in_raw_orientation_lands_in_working_frame(tmp_path):
    raw = smooth_image(1000, 600, seed=3)              # stored landscape, displayed portrait (EXIF 6)
    gt_raw = np.zeros((600, 1000), np.uint8)
    gt_raw[100:200, 0:300] = 255                       # top-left of the stored frame
    item = PairItem(
        "r",
        before_path=save(tmp_path / "b.jpg", raw, 6),
        after_path=save(tmp_path / "a.jpg", raw, 6),
        mask_path=save(tmp_path / "m.png", gt_raw),    # mask in the raw (untransposed) frame
    )
    pair = load_pair(item, work_long=256)
    assert pair.after.shape == (500, 300, 3) and pair.gt_mask.shape == (500, 300)
    expected_full = np.rot90(gt_raw > 127, -1)
    expected = cv2.resize(expected_full.astype(np.uint8), (300, 500), interpolation=cv2.INTER_NEAREST) > 0
    np.testing.assert_array_equal(pair.gt_mask, expected)
    assert pair.meta["orig_size"] == (600, 1000)


def test_load_pair_rejects_bad_work_long(tmp_path):
    img = save(tmp_path / "a.png", smooth_image(300, 200))
    with pytest.raises(CywsError, match="work_long"):
        load_pair(PairItem("x", img, img), work_long=-5)


# ------------------------------------------------------------------ box_orig

def test_detection_to_dict_box_orig():
    d = Detection(10, 20, 30, 40, 0.9, "after", label="appeared")
    assert "box_orig" not in d.to_dict()
    assert d.to_dict((2.0, 2.5))["box_orig"] == [20.0, 50.0, 60.0, 100.0]
    assert d.to_dict((2.0, 2.5))["box"] == [10.0, 20.0, 30.0, 40.0]


def _result(meta: dict, h: int = 100, w: int = 120) -> Result:
    det = Detection(10.0, 20.0, 50.5, 60.25, 0.8, "after", label="appeared")
    return Result(
        pair_id="s", detections=[det], mask=np.zeros((h, w), bool), heat=np.zeros((25, 30), np.float32),
        windows=[Window(0, 0, 0, 0, w, h, ALL_EDGES)], registration=None, timings_ms={"total": 1.0}, meta=meta,
    )


def test_save_outputs_box_orig_scaling(tmp_path):
    after = np.zeros((100, 120, 3), np.uint8)
    paths = post_process.save_outputs(_result({"scale": [2.0, 2.0], "orig_size": [240, 200]}), after, tmp_path)
    doc = json.loads(Path(paths["json"]).read_text())
    det = doc["detections"][0]
    assert det["box"] == [10.0, 20.0, 50.5, 60.25]
    assert det["box_orig"] == [20.0, 40.0, 101.0, 120.5]
    assert doc["image_size"] == {"w": 120, "h": 100} and doc["orig_image_size"] == {"w": 240, "h": 200}
    assert doc["scale"] == [2.0, 2.0]
    assert cv2.imread(paths["overlay"]).shape == (100, 120, 3)  # overlays at working resolution

    # No scale recorded (hand-built result): box_orig == box.
    doc = json.loads(Path(post_process.save_outputs(_result({}), after, tmp_path / "n")["json"]).read_text())
    assert doc["detections"][0]["box_orig"] == doc["detections"][0]["box"]
    assert doc["orig_image_size"] == {"w": 120, "h": 100}


def test_run_dataset_working_frame_end_to_end(tmp_path):
    """JPEG pair decoded at half resolution: boxes in the working frame, box_orig in the original."""
    import dataset_loader
    from app import CywsApp, run_dataset

    root = tmp_path / "ds"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    before = smooth_image(1280, 960, seed=5)
    after = before.copy()
    after[400:640, 600:900] = (250, 30, 30)            # a big red block appears
    gt = np.zeros((960, 1280), np.uint8)
    gt[400:640, 600:900] = 255
    save(root / "images" / "p1_ref.jpg", before)
    save(root / "images" / "p1_cap.jpg", after)
    save(root / "masks" / "p1.png", gt)
    items, _ = dataset_loader.discover_pairs(root)

    be = backends.FakeBackend()
    app = CywsApp(be, AppConfig(levels=(0, 1), work_long=512))
    pair = app.load(items[0])
    assert pair.after.shape == (480, 640, 3) and pair.meta["scale"] == pytest.approx((2.0, 2.0))
    out = tmp_path / "out"
    summary = run_dataset(app, items, out, log=lambda s: None)
    assert summary["n_ok"] == 1
    doc = json.loads((out / "p1.json").read_text())
    assert doc["image_size"] == {"w": 640, "h": 480} and doc["orig_image_size"] == {"w": 1280, "h": 960}
    assert doc["detections"], "the fake backend must see the block"
    for det in doc["detections"]:
        assert det["box_orig"] == pytest.approx([2.0 * v for v in det["box"]], abs=0.011)
    best = max(doc["detections"], key=lambda d: d["score"])
    x1, y1, x2, y2 = best["box_orig"]
    assert 550 < (x1 + x2) / 2 < 950 and 350 < (y1 + y2) / 2 < 690
    with (out / "summary.csv").open() as f:
        row = next(csv.DictReader(f))
    assert (row["width"], row["height"], row["orig_width"], row["orig_height"]) == ("640", "480", "1280", "960")
    assert float(row["recall"]) > 0.3 and "mask_read_ms" in row
    be.close()


# ------------------------------------------------------------------ threaded crops

class RecordingBackend(backends.Backend):
    """Async-style backend (submit queues, collect computes) that records every input."""

    name = "recording"

    def __init__(self, max_in_flight: int = 2, delay_s: float = 0.0) -> None:
        super().__init__(max_in_flight)
        self.delay_s = delay_s
        self.seen: list[tuple] = []
        self.streams: list[list] = []
        self._q: list = []
        self._fake = backends.FakeBackend()

    def infer_stream(self, items):
        self.streams.append([])
        return super().infer_stream(items)

    def _submit(self, tag, before, after):
        self.seen.append((tag, before.copy(), after.copy()))
        self.streams[-1].append(tag)
        # A serial device: each inference starts when it is pushed and the previous one is done.
        now = time.perf_counter()
        self._busy_until = max(now, getattr(self, "_busy_until", 0.0)) + self.delay_s
        self._q.append((tag, before, after, self._busy_until))

    def _collect(self):
        tag, b, a, ready_at = self._q.pop(0)
        remaining = ready_at - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        return tag, self._fake._run(b, a)


def _scene(h: int = 720, w: int = 960):
    before = smooth_image(w, h, seed=7)
    after = before.copy()
    after[300:420, 500:640] = (240, 20, 20)
    return before, after


@pytest.mark.parametrize("threads", [1, 4, 8])
def test_threaded_crops_keep_order_tags_and_contents(threads):
    from app import CywsApp

    before, after = _scene()
    pair = Pair("thr", before, after, meta={})
    be = RecordingBackend(max_in_flight=2)
    cfg = AppConfig(levels=(0, 1, 2), tile=256, tiles="full", preprocess_threads=threads)
    res = CywsApp(be, cfg).process(pair)
    tags = [t for t, _, _ in be.seen]
    assert tags == [("thr", w.id) for w in res.windows]
    assert [w.id for w in res.windows] == sorted(w.id for w in res.windows) and len(res.windows) > 10
    by_id = {w.id: w for w in res.windows}
    for (_, wid), tb, ta in be.seen:
        rb, ra = crop_pair(before, after, by_id[wid])
        np.testing.assert_array_equal(tb, rb)
        np.testing.assert_array_equal(ta, ra)

    # Same detections as cropping inline in the main thread.
    be0 = RecordingBackend(max_in_flight=2)
    res0 = CywsApp(be0, AppConfig(levels=(0, 1, 2), tile=256, tiles="full", preprocess_threads=0)).process(
        Pair("thr", before, after, meta={}))
    assert [d.to_dict() for d in res.detections] == [d.to_dict() for d in res0.detections]
    assert [t for t, _, _ in be0.seen] == tags


def test_crops_really_run_in_worker_threads(monkeypatch):
    from app import CywsApp

    names = set()
    orig = ingestion.crop_pair

    def spy(*a, **k):
        names.add(threading.current_thread().name)
        return orig(*a, **k)

    monkeypatch.setattr(ingestion, "crop_pair", spy)
    before, after = _scene()
    CywsApp(backends.FakeBackend(), AppConfig(levels=(0, 1), preprocess_threads=3)).process(
        Pair("w", before, after, meta={}))
    assert names and all(n.startswith("cyws-crop") for n in names)
    names.clear()
    CywsApp(backends.FakeBackend(), AppConfig(levels=(0, 1), preprocess_threads=0)).process(
        Pair("w", before, after, meta={}))
    assert names == {threading.main_thread().name}


def test_gated_mode_streams_level0_first_with_threads():
    from app import CywsApp

    before, after = _scene()
    be = RecordingBackend(max_in_flight=2)
    res = CywsApp(be, AppConfig(levels=(0, 1), tiles="gated", preprocess_threads=4)).process(
        Pair("g", before, after, meta={}))
    levels = {w.id: w.level for w in res.windows}
    assert len(be.streams) == 2
    assert all(levels[t[1]] == 0 for t in be.streams[0])
    assert be.streams[1] and all(levels[t[1]] == 1 for t in be.streams[1])
    assert [t for s in be.streams for t in s] == [("g", w.id) for w in res.windows]


def test_crop_error_in_worker_propagates_and_backend_recovers(monkeypatch):
    from app import CywsApp

    orig = ingestion.crop_pair
    calls = {"n": 0}

    def flaky(before, after, win, interp="area"):
        calls["n"] += 1
        if win.id == 3:
            raise CywsError("bad crop")
        return orig(before, after, win, interp)

    monkeypatch.setattr(ingestion, "crop_pair", flaky)
    before, after = _scene()
    be = backends.FakeBackend()
    app = CywsApp(be, AppConfig(levels=(0, 1), tile="auto", tiles="full", preprocess_threads=4))
    with pytest.raises(CywsError, match="bad crop"):
        app.process(Pair("e", before, after, meta={}))
    monkeypatch.setattr(ingestion, "crop_pair", orig)
    assert app.process(Pair("e", before, after, meta={})).windows  # backend left in sync


# ------------------------------------------------------------------ timings

@pytest.mark.parametrize("threads", [0, 4])
def test_timings_add_up_and_infer_wait_is_real(threads):
    from app import TIMING_KEYS, CywsApp

    before, after = _scene()
    delay = 0.02
    be = RecordingBackend(max_in_flight=2, delay_s=delay)
    res = CywsApp(be, AppConfig(levels=(0, 1), preprocess_threads=threads)).process(
        Pair("t", before, after, meta={"load_ms": 5.0}))
    t = res.timings_ms
    assert tuple(t) == TIMING_KEYS
    assert set(t) >= {"load", "register", "stream", "preprocess_cpu", "infer_wait", "postprocess", "total"}
    assert all(v >= 0 for v in t.values())
    assert t["load"] == 5.0
    assert t["total"] == pytest.approx(t["load"] + t["register"] + t["stream"] + t["postprocess"], abs=0.01)
    inner = t["preprocess_wait"] + t["infer_wait"] + t["decode"]
    assert inner <= t["stream"] + 0.05 and inner >= 0.8 * t["stream"]
    n = len(res.windows)
    # The device needs `delay` per window and nothing hides it here (crops are fast),
    # so the stream lasts >= n * delay and most of that is waiting on the backend.
    assert t["stream"] >= n * delay * 1e3 * 0.95
    assert t["infer_wait"] >= 0.5 * n * delay * 1e3
    assert t["preprocess_cpu"] > 0
    lat = be.stats()["latency"]
    assert lat["p50_ms"] >= delay * 1e3 * 0.95  # push -> output, not the 0.0 ms queue hand-off


class SlowSync(backends.FakeBackend):
    def _run(self, before, after):
        time.sleep(0.015)
        return super()._run(before, after)


def test_sync_backend_latency_is_compute_time():
    be = SlowSync(max_in_flight=2)
    rng = np.random.default_rng(0)
    items = [(i, rng.random((256, 256, 3), dtype=np.float32), rng.random((256, 256, 3), dtype=np.float32))
             for i in range(5)]
    for _tag, _out in be.infer_stream(items):
        time.sleep(0.03)  # slow consumer: must not inflate the compute latency
    st = be.stats()
    assert st["inferences"] == 5
    assert 14.0 <= st["latency"]["p50_ms"] < 28.0
    assert st["latency"]["max_ms"] < 30.0 + 15.0
    assert set(st["collect_wait"]) >= {"mean_ms", "p50_ms"}
    assert st["collect_wait"]["max_ms"] < 5.0  # results were already computed at push time


def test_async_latency_measured_from_push():
    be = RecordingBackend(max_in_flight=1, delay_s=0.02)
    rng = np.random.default_rng(0)
    items = [(i, rng.random((256, 256, 3), dtype=np.float32), rng.random((256, 256, 3), dtype=np.float32))
             for i in range(4)]
    assert [t for t, _ in be.infer_stream(items)] == [0, 1, 2, 3]
    st = be.stats()
    assert st["latency"]["p50_ms"] >= 19.0
    assert st["collect_wait"]["p50_ms"] >= 15.0  # the caller blocked in collect for the device


# ------------------------------------------------------------------ config / CLI

@pytest.mark.parametrize("kwargs", [dict(work_long=100), dict(work_long=-1), dict(work_long=1.5),
                                    dict(preprocess_threads=-1), dict(preprocess_threads=100)])
def test_invalid_speed_config_rejected(kwargs):
    from app import CywsApp

    with pytest.raises(CywsError):
        CywsApp(backends.FakeBackend(), AppConfig(**kwargs))


def test_defaults_and_cli_mapping():
    import run_cyws_neat

    cfg = AppConfig()
    assert cfg.work_long == 1920 and cfg.preprocess_threads == 4
    args = run_cyws_neat.build_parser().parse_args(["--work-long", "0", "--threads", "2"])
    cfg = run_cyws_neat.config_from_args(args)
    assert cfg.work_long == 0 and cfg.preprocess_threads == 2
    args = run_cyws_neat.build_parser().parse_args([])
    assert run_cyws_neat.config_from_args(args).work_long == 1920
    with pytest.raises(SystemExit):
        run_cyws_neat.build_parser().parse_args(["--threads", "-1"])


def test_stage_table_mentions_new_keys():
    import run_cyws_neat

    row = {"status": "ok", "windows": 13, **{f"t_{k}_ms": 1.0 for k in
           ("load", "register", "stream", "preprocess_cpu", "preprocess_wait", "infer_wait", "decode",
            "postprocess", "total")}}
    text = run_cyws_neat._stage_table({"pairs": [row], "backend_stats": {"latency": {"mean_ms": 35.0}}})
    for k in ("stream", "preprocess_cpu", "infer_wait", "backend latency"):
        assert k in text
