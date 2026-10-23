"""Model families: what the web app needs to load, configure and run each kind of model.

One object per family, with the same small interface:

    name, title, option_spec, capabilities
    presets() -> {name: {option: value}}       defaults() -> {option: value}
    base_config()                              config_from(base, opts) -> config
    open(path, backend_kind, onnx) -> backend  warm(backend)
    load(item, cfg) -> Pair                    process(backend, pair, cfg) -> common.Result

`cyws` wraps the CYWS app as it is (CywsApp, AppConfig, PRESETS, the CYWS backends);
`dinov2` is the zero-shot DINOv2 model (zeroshot.py, feature_backends.py). Nothing here
imports pyneat; the neat backends import it when they are built.
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

import numpy as np

import backends
import common
from app import CywsApp
from common import AppConfig, CywsError, Pair, PairItem, Result

ROOT = Path(__file__).resolve().parent.parent

# Run settings the page may change, with their allowed values (anything else is rejected).
CYWS_OPTION_SPEC = {
    "thr": (float, 0.05, 0.95),
    "levels": (str, ("0", "0,1", "0,1,2")),
    "register": (bool,),
    "tiles": (str, ("full", "gated")),
    "mask_mode": (str, ("box", "diff")),
    "verify_change": (bool,),
}
DINOV2_OPTION_SPEC = {
    "thr": (float, 0.10, 0.90),
    "levels": (str, ("0", "0,1", "0,1,2")),
    "register": (bool,),
    "tiles": (str, ("full", "gated")),
    "mask_mode": (str, ("box", "diff")),       # accepted and ignored: the mask is per pixel
    "verify_change": (bool,),                  # accepted and ignored
}


def _listify(v: Any) -> Any:
    return list(v) if isinstance(v, tuple) else v


def spec_json(spec: dict, ignored: tuple[str, ...] = ()) -> dict:
    """An option spec as JSON: {type, range | choices[, ignored]}."""
    out = {}
    for k, s in spec.items():
        if s[0] is bool:
            row = {"type": "bool"}
        elif s[0] is float:
            row = {"type": "float", "range": [s[1], s[2]]}
        else:
            row = {"type": "choice", "choices": list(s[1])}
        if k in ignored:
            row["ignored"] = True
        out[k] = row
    return out


def apply_options(cfg: Any, spec: dict, opts: dict, presets: dict) -> tuple[Any, dict]:
    """cfg with a preset and the checked options applied; also returns the option updates."""
    fields = {f.name for f in dataclasses.fields(cfg)}
    preset = opts.get("preset")
    if preset:
        p = presets.get(preset)
        if p is None:
            raise CywsError(f"unknown preset {preset!r}")
        cfg = dataclasses.replace(cfg, **{k: (tuple(v) if isinstance(v, list) else v)
                                          for k, v in p.items() if k in fields})
    upd = {}
    for k, s in spec.items():
        if k not in opts or opts[k] is None:
            continue
        v = opts[k]
        if s[0] is bool:
            upd[k] = bool(v)
        elif s[0] is float:
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise CywsError(f"{k.capitalize()} must be a number") from None
            if not s[1] <= v <= s[2]:
                raise CywsError(f"{k} must be in [{s[1]}, {s[2]}]")
            upd[k] = v
        else:
            if str(v) not in s[1]:
                raise CywsError(f"{k} must be one of {s[1]}")
            upd[k] = tuple(int(x) for x in str(v).split(",")) if k == "levels" else str(v)
    return cfg, upd


class Family:
    name = ""
    title = ""
    option_spec: dict = {}
    ignored: tuple[str, ...] = ()
    capabilities: dict = {}

    def presets(self) -> dict:
        raise NotImplementedError

    def base_config(self) -> Any:
        raise NotImplementedError

    def defaults(self, base: Any = None) -> dict:
        base = self.base_config() if base is None else base
        return {k: _listify(getattr(base, k)) for k in self.option_spec if hasattr(base, k)}

    def describe(self) -> dict:
        """JSON-able summary for /api/status."""
        return {"title": self.title, "presets": self.presets(), "defaults": self.defaults(),
                "option_spec": spec_json(self.option_spec, self.ignored), "capabilities": dict(self.capabilities)}


class CywsFamily(Family):
    """CYWS (trained): two 256 x 256 windows in, CenterNet boxes for both branches out."""

    name, title = "cyws", "CYWS"
    option_spec = CYWS_OPTION_SPEC
    capabilities = {"per": "window", "pixel_masks": True, "labels": True}

    def _cli_presets(self) -> dict | None:
        import run_cyws_neat as cli
        p = getattr(cli, "PRESETS", None)
        return p if isinstance(p, dict) else None

    def presets(self) -> dict:
        try:
            p = self._cli_presets()
            if p is not None:
                return {k: {kk: _listify(vv) for kk, vv in v.items()} for k, v in p.items()}
        except Exception:
            pass
        return {}

    def base_config(self) -> AppConfig:
        """AppConfig with the CLI's 'balanced' preset applied when the CLI defines presets."""
        cfg = AppConfig()
        try:
            presets = self._cli_presets()
            if presets is not None and "balanced" in presets:
                fields = {f.name for f in dataclasses.fields(AppConfig)}
                cfg = dataclasses.replace(cfg, **{k: v for k, v in presets["balanced"].items() if k in fields})
        except Exception:      # presets are a convenience; the dataclass defaults are fine
            pass
        return cfg

    def config_from(self, base: AppConfig, opts: dict) -> AppConfig:
        cfg, upd = apply_options(base, self.option_spec, opts, self.presets())
        if upd.get("mask_mode") == "diff" and not upd.get("register", cfg.register):
            upd["mask_mode"] = "box"          # diff masks need the aligned before image
        fields = {f.name for f in dataclasses.fields(AppConfig)}
        return dataclasses.replace(cfg, **{k: v for k, v in upd.items() if k in fields})

    def open(self, path: str | Path, backend_kind: str, onnx: str = "") -> Any:
        if backend_kind == "neat":
            return backends.NeatBackend(str(path), max_in_flight=2)
        return backends.make_backend(backend_kind, onnx)

    def warm(self, backend: Any) -> None:
        z = np.zeros((common.MODEL_SIZE, common.MODEL_SIZE, 3), np.float32)
        backend.infer(z, z)

    def load(self, item: PairItem, cfg: AppConfig) -> Pair:
        return CywsApp(None, cfg).load(item)

    def process(self, backend: Any, pair: Pair, cfg: AppConfig) -> Result:
        return CywsApp(backend, cfg).process(pair)


class Dinov2Family(Family):
    """DINOv2 ViT-S/14 (zero-shot): one 448 x 448 image in, patch features out."""

    name, title = "dinov2", "DINOv2"
    option_spec = DINOV2_OPTION_SPEC
    ignored = ("mask_mode", "verify_change")
    capabilities = {"per": "image", "pixel_masks": False, "labels": True}
    PRESETS = {
        "fast": {"levels": [0], "thr": 0.40},
        "balanced": {"levels": [0, 1], "tiles": "gated", "thr": 0.40},
        "accurate": {"levels": [0, 1], "tiles": "full", "thr": 0.35},
    }

    def presets(self) -> dict:
        return {k: dict(v) for k, v in self.PRESETS.items()}

    def base_config(self):
        from zeroshot import ZeroShotConfig
        return ZeroShotConfig()

    def config_from(self, base, opts: dict):
        cfg, upd = apply_options(base, self.option_spec, opts, self.presets())
        return dataclasses.replace(cfg, **upd)

    @staticmethod
    def onnx_path(onnx: str = "") -> Path:
        """A DINOv2 ONNX for --backend onnx: DINOV2_ONNX, --onnx when it is a DINOv2 file, else build/onnx."""
        env = os.environ.get("DINOV2_ONNX", "").strip()
        if env:
            return Path(env)
        if onnx and "dinov2" in Path(onnx).name:
            return Path(onnx)
        return ROOT / "build/onnx/dinov2_vits14_reg_448.onnx"

    def open(self, path: str | Path, backend_kind: str, onnx: str = "") -> Any:
        import feature_backends as fb
        if backend_kind == "neat":
            return fb.NeatFeatureBackend(path)
        if backend_kind == "fake":
            return fb.FakeFeatureBackend()
        if backend_kind == "onnx":
            return fb.OnnxFeatureBackend(self.onnx_path(onnx))
        raise CywsError(f"unknown backend {backend_kind!r} for DINOv2")

    def warm(self, backend: Any) -> None:
        backend.features(np.zeros((backend.size, backend.size, 3), np.float32))

    def load(self, item: PairItem, cfg) -> Pair:
        from zeroshot import ZeroShotApp
        return ZeroShotApp(None, cfg).load(item)

    def process(self, backend: Any, pair: Pair, cfg) -> Result:
        from zeroshot import ZeroShotApp
        return ZeroShotApp(backend, cfg).process(pair)


FAMILIES: dict[str, Family] = {f.name: f for f in (CywsFamily(), Dinov2Family())}


def get(name: str) -> Family:
    fam = FAMILIES.get(name or "cyws")
    if fam is None:
        raise CywsError(f"unknown model family {name!r}")
    return fam
