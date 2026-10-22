"""Cross-module integration checks added while wiring the app together.

They pin the fixes made at integration time: the fast diff helpers must match
the straightforward numpy definitions, --save-heat must reach save_outputs, and
the verbose static run must find the JSON even when the pair id is sanitised.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import post_process
import registration
import run_cyws_neat
import tiling
from common import AppConfig, Pair, Window
from test_app import make_scene, paste_change


def _reference_energy(a: np.ndarray, b: np.ndarray, valid: np.ndarray, s: int) -> np.ndarray:
    d = registration.abs_diff_gray(a, b)
    d[~valid] = 0
    h, w = d.shape
    gh, gw = -(-h // s), -(-w // s)
    pad = np.zeros((gh * s, gw * s))
    cnt = np.zeros_like(pad)
    pad[:h, :w] = d
    cnt[:h, :w] = 1
    return pad.reshape(gh, s, gw, s).sum((1, 3)) / cnt.reshape(gh, s, gw, s).sum((1, 3))


@pytest.mark.parametrize("hw", [(1001, 1503), (64, 64), (70, 93), (257, 256)])
@pytest.mark.parametrize("scale", [1, 3, 4])
def test_diff_energy_map_matches_reference(hw, scale):
    rng = np.random.default_rng(hw[0] * 7 + scale)
    a = rng.integers(0, 256, (*hw, 3), dtype=np.uint8)
    b = rng.integers(0, 256, (*hw, 3), dtype=np.uint8)
    valid = rng.random(hw) > 0.2
    got = registration.diff_energy_map(a, b, valid, scale)
    ref = _reference_energy(a, b, valid, scale)
    assert got.dtype == np.float32 and got.shape == ref.shape
    np.testing.assert_allclose(got, ref, atol=1e-3)


def test_median_abs_diff_matches_numpy():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)
    b = np.clip(a.astype(int) + rng.integers(-20, 21, a.shape), 0, 255).astype(np.uint8)
    valid = rng.random((300, 400)) > 0.3
    ref = float(np.median(registration.abs_diff_gray(a, b)[valid]))
    assert abs(registration.median_abs_diff(a, b, valid) - ref) <= 0.5
    assert abs(registration.median_abs_diff(a, b) - float(np.median(registration.abs_diff_gray(a, b)))) <= 0.5
    assert registration.median_abs_diff(a, b, np.zeros((300, 400), bool)) == 0.0


def test_filter_window_uses_tiling_mapping():
    win = Window(id=3, level=1, x0=100, y0=50, w=512, h=512, img_edges=(False, False, False, False))
    dets = np.array([[64.0, 64.0, 128.0, 160.0, 0.9]], np.float32)
    out = post_process.filter_window(dets, win, "after", AppConfig())
    want = tiling.window_to_canonical(dets[:, :4], win)[0]
    assert len(out) == 1
    np.testing.assert_allclose(out[0].box, want)
    assert out[0].window_id == 3 and out[0].level == 1


def _write_pair(tmp: Path, stem_after: str) -> tuple[Path, Path]:
    before = make_scene(1)
    after = paste_change(before)
    b, a = tmp / "before.png", tmp / f"{stem_after}.png"
    Image.fromarray(before).save(b)
    Image.fromarray(after).save(a)
    return b, a


def test_cli_save_heat_and_verbose_with_unsafe_id(tmp_path, capsys):
    b, a = _write_pair(tmp_path, "after shot#1")
    out = tmp_path / "out"
    rc = run_cyws_neat.main([
        "--mode", "static", "--backend", "fake", "--before", str(b), "--after", str(a),
        "--out", str(out), "--levels", "0,1", "--save-heat", "-v",
    ])
    assert rc == 0
    heat = list(out.glob("*_heat.png"))
    assert len(heat) == 1 and heat[0].name == "after_shot_1_heat.png"
    printed = capsys.readouterr().out
    assert '"label"' in printed  # the detections JSON was found and echoed
    doc = json.loads((out / "after_shot_1.json").read_text())
    assert doc["num_detections"] >= 1 and "heat" in doc["files"]


def test_cli_without_save_heat_writes_no_heat(tmp_path):
    b, a = _write_pair(tmp_path, "after")
    out = tmp_path / "out"
    assert run_cyws_neat.main(["--mode", "static", "--backend", "fake", "--before", str(b),
                               "--after", str(a), "--out", str(out), "--levels", "0"]) == 0
    assert not list(out.glob("*_heat.png"))


def test_verify_skipped_without_detections():
    import backends
    from app import CywsApp

    before = make_scene(2)
    app = CywsApp(backends.make_backend("fake"), AppConfig(levels=(0,), register=True))
    res = app.process(Pair(pair_id="same", before=before.copy(), after=before.copy()))
    assert res.detections == []
    assert res.meta.get("verify_dropped") == []
