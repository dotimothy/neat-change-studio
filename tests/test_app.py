"""End-to-end tests of CywsApp, run_dataset and the CLI with the FakeBackend.

The synthetic pair is a textured scene (so SIFT registration has features) with
a rectangle pasted into the after image; every pipeline variant must put a box
and mask pixels on that rectangle.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from common import AppConfig, CywsError, Pair, PairItem, Result

APP_DIR = Path(__file__).resolve().parents[1] / "app"
H, W = 480, 640
PASTE = (380, 120, 500, 230)  # x1, y1, x2, y2 in the after frame


def make_scene(seed: int = 0, h: int = H, w: int = W) -> np.ndarray:
    """Smooth colour noise plus random shapes: textured enough for SIFT."""
    import cv2

    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
    img = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    for _ in range(60):
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        if rng.random() < 0.5:
            cv2.circle(img, (x, y), int(rng.integers(5, 30)), color, -1)
        else:
            cv2.rectangle(img, (x, y), (x + int(rng.integers(8, 50)), y + int(rng.integers(8, 50))), color, -1)
    return np.ascontiguousarray(img)


def paste_change(before: np.ndarray, box=PASTE) -> np.ndarray:
    """Paste a high-contrast checkerboard object into a copy of the scene."""
    x1, y1, x2, y2 = box
    after = before.copy()
    yy, xx = np.mgrid[y1:y2, x1:x2]
    checker = ((xx // 12 + yy // 12) % 2).astype(bool)
    patch = np.where(checker[..., None], np.array([250, 20, 20], np.uint8), np.array([20, 20, 250], np.uint8))
    after[y1:y2, x1:x2] = patch
    return after


def gt_mask(box=PASTE, h: int = H, w: int = W) -> np.ndarray:
    m = np.zeros((h, w), bool)
    x1, y1, x2, y2 = box
    m[y1:y2, x1:x2] = True
    return m


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def hits_paste(result: Result, box=PASTE, margin: float = 15.0) -> bool:
    """True if a detection overlaps the pasted rectangle or is centred on it."""
    x1, y1, x2, y2 = box
    for d in result.detections:
        cx, cy = (d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2
        if box_iou((d.x1, d.y1, d.x2, d.y2), box) >= 0.2:
            return True
        if x1 - margin <= cx <= x2 + margin and y1 - margin <= cy <= y2 + margin:
            return True
    return False


@pytest.fixture(scope="module")
def scene() -> tuple[np.ndarray, np.ndarray]:
    before = make_scene(0)
    return before, paste_change(before)


@pytest.fixture()
def fake_backend():
    import backends

    be = backends.make_backend("fake")
    yield be
    be.close()


def make_pair(before: np.ndarray, after: np.ndarray, pair_id: str = "synth") -> Pair:
    return Pair(pair_id=pair_id, before=before.copy(), after=after.copy(), gt_mask=gt_mask(), meta={})


# -- CywsApp -------------------------------------------------------------------------


@pytest.mark.parametrize("levels", [(0,), (0, 1)])
@pytest.mark.parametrize("tiles", ["full", "gated"])
@pytest.mark.parametrize("register", [False, True])
@pytest.mark.parametrize("mask_mode", ["box", "diff"])
def test_process_detects_pasted_change(scene, fake_backend, levels, tiles, register, mask_mode):
    from app import CywsApp

    before, after = scene
    cfg = AppConfig(levels=levels, tiles=tiles, register=register, mask_mode=mask_mode)
    res = CywsApp(fake_backend, cfg).process(make_pair(before, after))

    assert isinstance(res, Result) and res.pair_id == "synth"
    assert hits_paste(res), [d.to_dict() for d in res.detections]
    assert all(d.label in ("appeared", "disappeared", "changed") for d in res.detections)

    assert res.mask.shape == (H, W) and res.mask.dtype == bool
    x1, y1, x2, y2 = PASTE
    assert res.mask[y1:y2, x1:x2].mean() > 0.3
    assert res.mask.sum() < 0.5 * H * W  # the change is local, not the whole frame

    assert res.heat.shape == (math.ceil(H / 4), math.ceil(W / 4)) and res.heat.dtype == np.float32
    assert res.heat[y1 // 4 : y2 // 4, x1 // 4 : x2 // 4].max() > 0.3

    assert res.windows and res.windows[0].level == 0
    assert {w.level for w in res.windows} <= set(levels)
    assert [w.level for w in res.windows] == sorted(w.level for w in res.windows)  # level 0 first
    if levels == (0, 1) and tiles == "full":
        assert any(w.level == 1 for w in res.windows)
    if tiles == "gated" and levels == (0, 1):
        assert "windows_gated_out" in res.meta
        assert res.meta["windows_run"] == len(res.windows) <= res.meta["windows_planned"]
        # level-0 heat sits on the paste, so the windows covering it must survive gating
        assert any(w.level == 1 and w.x0 < x2 and w.x1 > x1 and w.y0 < y2 and w.y1 > y1 for w in res.windows)

    t = res.timings_ms
    assert set(t) >= {"load", "register", "stream", "preprocess_cpu", "infer_wait", "postprocess", "total"}
    assert all(v >= 0 for v in t.values())
    assert t["total"] == pytest.approx(t["load"] + t["register"] + t["stream"] + t["postprocess"], abs=0.01)

    assert res.before_aligned is not None and res.before_aligned.shape == after.shape
    if register:
        assert res.registration is not None
        assert res.registration.valid_mask.shape == (H, W)
    else:
        assert res.registration is None
    json.dumps(res.meta)  # meta must be JSON-serialisable for save_outputs


def test_identical_images_give_no_change(scene, fake_backend):
    from app import CywsApp

    before, _ = scene
    res = CywsApp(fake_backend, AppConfig(levels=(0, 1))).process(make_pair(before, before, "same"))
    assert res.detections == []
    assert not res.mask.any()


@pytest.mark.parametrize("levels", [(0,), (0, 1), (0, 1, 2)])
def test_object_across_level0_seam_is_one_box(fake_backend, levels):
    """1920x1080 gets two level-0 squares; an object over their seam must not split."""
    from app import CywsApp

    before = np.full((1080, 1920, 3), 90, np.uint8)
    after = before.copy()
    after[300:700, 700:1300] = (240, 30, 30)
    pair = Pair(pair_id="wide", before=before, after=after, gt_mask=None, meta={})
    res = CywsApp(fake_backend, AppConfig(levels=levels)).process(pair)
    assert len(res.detections) == 1, [d.to_dict() for d in res.detections]
    d = res.detections[0]
    assert d.label == "changed"
    assert box_iou((d.x1, d.y1, d.x2, d.y2), (700, 300, 1300, 700)) > 0.9


def test_photometric_and_wbf_variants_run(scene, fake_backend):
    from app import CywsApp

    before, after = scene
    cfg = AppConfig(levels=(0, 1), register=True, photometric=True, merge="wbf", resize="bilinear",
                    verify_change=False)
    res = CywsApp(fake_backend, cfg).process(make_pair(before, after))
    assert hits_paste(res)
    assert "verify_dropped" not in res.meta


def test_native_size_before_is_resized_without_registration(scene, fake_backend):
    """A hand-built Pair with a differently sized before still lands in the canonical frame."""
    import cv2
    from app import CywsApp

    before, after = scene
    small = cv2.resize(before, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    res = CywsApp(fake_backend, AppConfig(levels=(0,))).process(make_pair(small, after))
    assert res.before_aligned.shape == after.shape
    assert hits_paste(res)


@pytest.mark.parametrize(
    "kwargs",
    [dict(levels=()), dict(levels=(0, 3)), dict(thr=0.0), dict(overlap=1.0), dict(tiles="sparse"),
     dict(mask_mode="seg"), dict(tile=10)],
)
def test_invalid_config_rejected(fake_backend, kwargs):
    from app import CywsApp

    with pytest.raises(CywsError):
        CywsApp(fake_backend, AppConfig(**kwargs))


class _ReorderingBackend:
    """Returns outputs in reverse order to prove the FIFO tag check fires."""

    name = "reorder"

    def infer_stream(self, items):
        buf = [(tag, np.zeros((10, 256, 256), np.float32)) for tag, _b, _a in items]
        yield from reversed(buf)

    def close(self):
        pass


def test_out_of_order_backend_is_detected(scene):
    from app import CywsApp

    before, after = scene
    with pytest.raises(RuntimeError, match="FIFO"):
        CywsApp(_ReorderingBackend(), AppConfig(levels=(0, 1), tile="auto", tiles="full")).process(make_pair(before, after))


# -- run_dataset / CLI -------------------------------------------------------------------


def _save(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    """Tree with two good pairs (one without GT), one bad-aspect pair and an orphan."""
    root = tmp_path / "testCYWS"
    for i, pid in enumerate(("20240101_000001", "20240101_000002")):
        before = make_scene(10 + i)
        _save(root / "images" / f"{pid}_ref.png", before)
        _save(root / "images" / f"{pid}_cap.png", paste_change(before))
        if i == 0:
            _save(root / "masks" / f"{pid}.png", gt_mask().astype(np.uint8) * 255)
    bad = make_scene(20)
    _save(root / "images" / "20240101_000003_ref.png", make_scene(21, h=240, w=640))  # 8:3 vs 4:3
    _save(root / "images" / "20240101_000003_cap.png", paste_change(bad))
    _save(root / "images" / "20240101_000004_ref.png", make_scene(22))  # orphan: no _cap
    return root


def _check_outputs(out: Path, pair_ids: list[str]) -> None:
    for pid in pair_ids:
        for suffix in ("_overlay.jpg", "_mask.png", ".json"):
            f = out / f"{pid}{suffix}"
            assert f.is_file() and f.stat().st_size > 0, f
        info = json.loads((out / f"{pid}.json").read_text())
        assert "detections" in info


def test_run_dataset_writes_summary(dataset, tmp_path, fake_backend):
    import dataset_loader
    from app import CywsApp, run_dataset

    items, warnings = dataset_loader.discover_pairs(dataset)
    assert len(items) == 3 and warnings
    out = tmp_path / "out"
    logs: list[str] = []
    summary = run_dataset(CywsApp(fake_backend, AppConfig(levels=(0, 1))), items, out, log=logs.append)

    assert summary["n_pairs"] == 3 and summary["n_ok"] == 2 and summary["n_failed"] == 1
    assert summary["failed"][0]["pair_id"] == "20240101_000003"
    assert any("FAILED" in line for line in logs)
    _check_outputs(out, ["20240101_000001", "20240101_000002"])

    with (out / "summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [r["pair_id"] for r in rows] == ["20240101_000001", "20240101_000002", "20240101_000003", "mean"]
    assert rows[2]["status"] == "failed" and rows[2]["error"]
    assert int(rows[0]["n_boxes"]) >= 1 and float(rows[0]["t_total_ms"]) > 0

    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["n_ok"] == 2 and len(metrics["pairs"]) == 3
    assert metrics["config"]["levels"] == [0, 1]
    assert metrics["mean"]["pair_id"] == "mean" and metrics["mean"]["n_boxes"] >= 1
    assert metrics["pooled"]["pairs"] == 1  # only the first pair has GT
    assert metrics["pooled"]["iou"] > 0.1
    gt_row = metrics["pairs"][0]
    assert gt_row["recall"] > 0.3 and "f1" in gt_row


def test_run_dataset_writes_partial_summary_when_backend_dies(dataset, tmp_path, fake_backend, monkeypatch):
    """An unexpected exception mid-run still leaves summary.csv / metrics.json behind."""
    import dataset_loader
    from app import CywsApp, run_dataset

    items, _ = dataset_loader.discover_pairs(dataset)
    app = CywsApp(fake_backend, AppConfig(levels=(0,)))
    orig = app.process

    def dying_process(pair):
        if pair.pair_id == "20240101_000002":
            raise RuntimeError("infra.accelerator_execution_failed")
        return orig(pair)

    monkeypatch.setattr(app, "process", dying_process)
    out = tmp_path / "out"
    logs: list[str] = []
    with pytest.raises(RuntimeError, match="accelerator"):
        run_dataset(app, items, out, log=logs.append)
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["n_ok"] == 1 and "accelerator" in metrics["aborted"]
    assert [p["status"] for p in metrics["pairs"]] == ["ok", "failed"]
    with (out / "summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [r["pair_id"] for r in rows] == ["20240101_000001", "20240101_000002", "mean"]
    assert any("aborted" in line for line in logs)


def test_cli_dataset_mode_main(dataset, tmp_path, capsys):
    import run_cyws_neat

    out = tmp_path / "cli_out"
    rc = run_cyws_neat.main(["--mode", "dataset", "--dataset-dir", str(dataset), "--backend", "fake",
                             "--out", str(out), "--levels", "0,1", "--tiles", "gated", "--bench", "-v"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "warning:" in text and "done: 2/3" in text and "stage" in text
    _check_outputs(out, ["20240101_000001", "20240101_000002"])
    assert json.loads((out / "metrics.json").read_text())["config"]["tiles"] == "gated"


def test_cli_dataset_limit(dataset, tmp_path):
    import run_cyws_neat

    out = tmp_path / "lim"
    assert run_cyws_neat.main(["--mode", "dataset", "--dataset-dir", str(dataset), "--backend", "fake",
                               "--out", str(out), "--limit", "1", "--levels", "0"]) == 0
    assert json.loads((out / "metrics.json").read_text())["n_pairs"] == 1


def _run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(APP_DIR / "run_cyws_neat.py"), *args], cwd=cwd,
                          capture_output=True, text=True, timeout=300)


def test_cli_static_mode_subprocess(dataset, tmp_path):
    img = dataset / "images"
    out = tmp_path / "static_out"
    proc = _run_cli(["--mode", "static", "--before", str(img / "20240101_000001_ref.png"),
                     "--after", str(img / "20240101_000001_cap.png"),
                     "--mask", str(dataset / "masks" / "20240101_000001.png"),
                     "--backend", "fake", "--out", str(out), "--register", "--mask-mode", "diff",
                     "--save-heat", "-v"], cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    _check_outputs(out, ["20240101_000001_cap"])
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["n_ok"] == 1 and metrics["config"]["register"] is True
    assert metrics["config"]["verify_change"] is None and metrics["pooled"]["pairs"] == 1
    assert (out / "summary.csv").is_file()


@pytest.mark.parametrize(
    "args, message",
    [
        (["--mode", "static", "--after", "missing.jpg", "--before", "missing2.jpg", "--backend", "fake"], ""),
        (["--mode", "static", "--backend", "fake"], "--before and --after"),
        (["--mode", "dataset", "--backend", "fake"], "--dataset-dir"),
        (["--mode", "dataset", "--dataset-dir", "nowhere", "--backend", "fake"], ""),
        (["--mode", "static", "--before", "a.jpg", "--after", "b.jpg"], "--model-path"),
        (["--mode", "static", "--before", "a.jpg", "--after", "b.jpg", "--model-path", "m.onnx"], "model not found"),
    ],
)
def test_cli_rejects_bad_input_before_backend(tmp_path, args, message):
    out = tmp_path / "never"
    proc = _run_cli([*args, "--out", str(out)], cwd=tmp_path)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "error:" in proc.stderr and message in proc.stderr
    assert not out.exists()


def test_cli_bad_option_values():
    import run_cyws_neat

    for bad in (["--levels", "0,3"], ["--levels", "x"], ["--tile", "10"], ["--thr", "0"], ["--overlap", "1"],
                ["--backend", "tpu"]):
        with pytest.raises(SystemExit) as exc:
            run_cyws_neat.build_parser().parse_args(bad)
        assert exc.value.code == 2


def test_cli_config_mapping():
    import run_cyws_neat

    p = run_cyws_neat.build_parser()
    cfg = run_cyws_neat.config_from_args(p.parse_args(["--levels", "1,0", "--tile", "512", "--no-verify-change",
                                                       "--thr", "0.4", "--merge", "wbf", "--photometric"]))
    assert cfg.levels == (0, 1) and cfg.tile == 512 and cfg.verify_change is False
    assert cfg.thr == 0.4 and cfg.merge == "wbf" and cfg.photometric and not cfg.verify_enabled()
    assert run_cyws_neat.config_from_args(p.parse_args(["--register"])).verify_enabled()
    assert run_cyws_neat.config_from_args(p.parse_args(["--verify-change"])).verify_change is True


def test_backend_inferred_from_model_suffix(tmp_path):
    import run_cyws_neat

    p = run_cyws_neat.build_parser()
    onnx = tmp_path / "cyws.onnx"
    onnx.write_bytes(b"x")
    mpk = tmp_path / "cyws_bf16_mpk.tar.gz"
    mpk.write_bytes(b"x")
    assert run_cyws_neat.resolve_backend(p.parse_args(["--model-path", str(onnx)])) == "onnx"
    assert run_cyws_neat.resolve_backend(p.parse_args(["--model-path", str(mpk)])) == "neat"
    assert run_cyws_neat.resolve_backend(p.parse_args(["--backend", "fake"])) == "fake"
    with pytest.raises(CywsError):
        run_cyws_neat.resolve_backend(p.parse_args(["--backend", "neat", "--model-path", str(onnx)]))


def test_cli_imports_without_runtimes():
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import run_cyws_neat, app; "
        "bad = [m for m in ('pyneat', 'onnxruntime', 'afe') if m in sys.modules]; "
        "sys.exit(f'runtime imported at module load: {bad}' if bad else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code, str(APP_DIR)], capture_output=True, text=True, timeout=120,
                          env={**os.environ, "PYTHONPATH": ""})
    assert proc.returncode == 0, proc.stderr
