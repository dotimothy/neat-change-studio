"""Tuned defaults, --preset, fractional tiles and reproducible registration."""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np
import pytest

import tiling
from common import PRESETS, AppConfig, CywsError, preset_config


def _cli(*argv: str) -> AppConfig:
    import run_cyws_neat

    return run_cyws_neat.config_from_args(run_cyws_neat.build_parser().parse_args(list(argv)))


# -- defaults and presets ----------------------------------------------------------------

def test_defaults_are_the_tuned_balanced_preset():
    cfg = AppConfig()
    assert cfg.levels == (0, 1) and cfg.tile == 0.45 and cfg.tiles == "gated"
    assert cfg.gate_heat == 0.25 and cfg.thr == 0.29 and cfg.register is False
    assert preset_config("balanced") == cfg
    assert set(PRESETS) == {"fast", "balanced", "accurate"}
    # Every preset only names real fields and validates.
    from app import validate_config

    fields = {f.name for f in dataclasses.fields(AppConfig)}
    for name, over in PRESETS.items():
        assert set(over) <= fields, name
        validate_config(preset_config(name))


def test_preset_contents():
    fast, acc = preset_config("fast"), preset_config("accurate")
    assert fast.levels == (0,) and fast.thr == AppConfig().thr
    assert acc.levels == (0, 1) and acc.tiles == "full" and acc.tile == 0.45
    assert preset_config("fast", thr=0.4).thr == 0.4
    with pytest.raises(CywsError):
        preset_config("turbo")


def test_cli_default_is_balanced_and_matches_appconfig():
    assert _cli() == AppConfig()
    assert _cli("--preset", "balanced") == AppConfig()


def test_cli_preset_and_explicit_overrides():
    assert _cli("--preset", "fast").levels == (0,)
    cfg = _cli("--preset", "fast", "--levels", "0,1")
    assert cfg.levels == (0, 1) and cfg.thr == PRESETS["fast"]["thr"]
    assert _cli("--preset", "accurate").tiles == "full"
    assert _cli("--preset", "accurate", "--tiles", "gated").tiles == "gated"
    cfg = _cli("--preset", "accurate", "--thr", "0.35", "--tile", "auto", "--gate-heat", "0.1")
    assert (cfg.thr, cfg.tile, cfg.gate_heat, cfg.tiles) == (0.35, "auto", 0.1, "full")
    reg = _cli("--register")
    assert reg.register and reg.verify_enabled() and reg.levels == (0, 1)
    assert _cli("--register", "--no-register").register is False
    # Non-preset flags are untouched by the preset.
    assert _cli("--preset", "fast", "--overlap", "0.5", "--work-long", "0").overlap == 0.5


def test_cli_tile_parsing():
    import run_cyws_neat

    assert _cli("--tile", "0.3").tile == 0.3
    assert _cli("--tile", "512").tile == 512
    assert _cli("--tile", "1.0").tile == 1.0
    for bad in (["--tile", "1.5"], ["--tile", "0"], ["--tile", "0.0"], ["--tile", "-0.2"], ["--tile", "63"],
                ["--tile", "big"], ["--gate-heat", "1.5"], ["--preset", "turbo"]):
        with pytest.raises(SystemExit) as exc:
            run_cyws_neat.build_parser().parse_args(bad)
        assert exc.value.code == 2, bad


# -- fractional tiles --------------------------------------------------------------------

def test_resolve_tile():
    assert tiling.resolve_tile("auto", 1560, 2104) == "auto"
    assert tiling.resolve_tile(0.45, 1560, 2104) == 702
    assert tiling.resolve_tile(0.45, 1080, 1920) == 486
    assert tiling.resolve_tile(512, 1560, 2104) == 512
    assert tiling.resolve_tile(np.int64(300), 1560, 2104) == 300
    assert tiling.resolve_tile(0.01, 1000, 1000) == 64        # floor
    for bad in (True, 0.0, 1.5, -1, 0, "big"):
        with pytest.raises(CywsError):
            tiling.resolve_tile(bad, 1000, 1000)


@pytest.mark.parametrize("hw", [(1560, 2104), (1080, 1920), (3000, 4000), (2104, 1560)])
def test_fractional_tile_plans_like_pixels(hw):
    H, W = hw
    px = int(round(0.45 * min(H, W)))
    a = tiling.plan_windows(H, W, tile=0.45, levels=(0, 1))
    b = tiling.plan_windows(H, W, tile=px, levels=(0, 1))
    assert a == b and any(w.level == 1 for w in a)
    # The level-1 side scales with the frame, so a half-resolution frame tiles the same scene.
    half = tiling.plan_windows(H // 2, W // 2, tile=0.45, levels=(0, 1))
    assert len([w for w in half if w.level == 1]) == len([w for w in a if w.level == 1])


@pytest.mark.parametrize("kw", [dict(tile=1.5), dict(tile=0.0), dict(tile=True), dict(tile=32),
                                dict(gate_heat=1.5), dict(gate_heat=-0.1)])
def test_validate_rejects_bad_tile_and_gate(kw):
    from app import validate_config

    with pytest.raises(CywsError):
        validate_config(AppConfig(**kw))


def test_sweep_fraction_matches_app_resolution():
    """eval/sweep.py turns a tile fraction into pixels itself; it must agree with the app."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
    import sweep

    for h, w in ((1560, 2104), (1080, 1920), (1500, 2000)):
        spec = sweep.Spec("0,1", "0.45", "gated", 0.29, "off")
        assert sweep.make_cfg(AppConfig(), spec, h, w).tile == tiling.resolve_tile(0.45, h, w)


# -- reproducible registration -----------------------------------------------------------

def test_register_is_reproducible_regardless_of_rng_state():
    import registration as reg
    from test_registration import make_pair

    before, after, _ = make_pair()
    cv2.setRNGSeed(1)
    r1 = reg.register(before, after)
    # Disturb cv2's RNG the way unrelated code would.
    cv2.setRNGSeed(987654)
    noise = np.empty((64, 64), np.float32)
    for _ in range(7):
        cv2.randu(noise, 0, 1)
    r2 = reg.register(before, after)
    assert r1.ok and r2.ok
    assert np.array_equal(r1.H, r2.H) and r1.inliers == r2.inliers
