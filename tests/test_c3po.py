"""C-3PO in the web demo: the family, its backends (fake, mocked pyneat, ONNX), the app and the server."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("flask")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "webapp"))
import c3po_app  # noqa: E402
import c3po_backends as cb  # noqa: E402
import common  # noqa: E402
import families  # noqa: E402
import server  # noqa: E402
from common import CywsError, Pair  # noqa: E402
from test_families import make_demo, pasted_pair, upload  # noqa: E402
from test_speed import smooth_image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def demo(tmp_path):
    d = make_demo(tmp_path)
    d.load_model("c3po-pcd")
    yield d
    d.close()


# ---------------------------------------------------------------- family
def test_family_presets_options_and_capabilities():
    fam = families.get("c3po")
    assert isinstance(fam, families.C3POFamily) and fam.title == "C-3PO"
    assert fam.capabilities == {"per": "pair", "pixel_masks": False, "labels": False}
    assert set(fam.presets()) == {"fast", "balanced", "accurate"}
    assert fam.presets()["fast"] == {"levels": [0], "thr": 0.20}
    assert fam.defaults() == {"thr": 0.20, "levels": [0, 1], "register": False, "tiles": "gated",
                              "mask_mode": "box", "verify_change": None}
    for name, p in fam.presets().items():          # every preset is valid for the app
        cfg = fam.config_from(fam.base_config(), {"preset": name})
        c3po_app.validate_config(cfg)
        assert list(cfg.levels) == p["levels"] and cfg.thr == p["thr"]
    cfg = fam.config_from(fam.base_config(), {"preset": "accurate", "levels": "0,1,2", "mask_mode": "diff", "thr": 0.5})
    assert cfg.levels == (0, 1, 2) and cfg.tiles == "full" and cfg.thr == 0.5
    assert (cfg.gate, cfg.min_area, cfg.work_long) == (0.05, 0.002, 1024)
    for bad in ({"thr": 0.95}, {"thr": "x"}, {"levels": "1"}, {"tiles": "some"}, {"preset": "best"}):
        with pytest.raises(CywsError):
            fam.config_from(fam.base_config(), bad)
    spec = fam.describe()["option_spec"]
    assert spec["thr"] == {"type": "float", "range": [0.05, 0.90]}
    assert spec["mask_mode"]["ignored"] is True and spec["verify_change"]["ignored"] is True
    assert "ignored" not in spec["tiles"]
    json.dumps(fam.describe())


def test_checkpoint_and_onnx_path(monkeypatch, tmp_path):
    fam = families.C3POFamily
    assert fam.checkpoint("models/c3po_csb_256_bf16_mpk.tar.gz") == "csb"
    assert fam.checkpoint("c3po_pcd_256_bf16_mpk.tar.gz") == "pcd"
    assert fam.checkpoint("c3po-csb_mpk.tar.gz") == "csb"                 # any separator
    assert fam.checkpoint("CSB.onnx") == "csb"
    assert fam.checkpoint("something.tar.gz") is None                     # not a guess
    assert fam.checkpoint("pcdx_csb2.tar.gz") is None
    monkeypatch.delenv("C3PO_ONNX", raising=False)
    monkeypatch.setenv("C3PO_DIR", str(tmp_path / "C3PO"))
    assert fam.onnx_path("c3po_csb_256_bf16_mpk.tar.gz") in (
        families.ROOT / "build/onnx/c3po_csb_256.onnx", tmp_path / "C3PO/build/onnx/c3po_csb_256.onnx")
    assert fam.onnx_path("x", onnx="/m/c3po_pcd_256.onnx") == Path("/m/c3po_pcd_256.onnx")
    assert fam.onnx_path("c3po-pcd.tar.gz", onnx="build/onnx/cyws_fused.onnx").name == "c3po_pcd_256.onnx"
    with pytest.raises(CywsError, match="can't tell which C-3PO checkpoint"):
        fam.onnx_path("renamed.tar.gz", onnx="build/onnx/cyws_fused.onnx")
    monkeypatch.setenv("C3PO_ONNX", "/env/c3po.onnx")
    assert fam.onnx_path("c3po_csb_256_bf16_mpk.tar.gz") == Path("/env/c3po.onnx")
    with pytest.raises(CywsError, match="ONNX model not found"):
        families.get("c3po").open("c3po_pcd_256_bf16_mpk.tar.gz", "onnx")
    with pytest.raises(CywsError, match="unknown backend"):
        families.get("c3po").open("x", "tpu")


# ---------------------------------------------------------------- backends
def test_change_logit_is_one_minus_p_no_change():
    rng = np.random.default_rng(0)
    for c in (2, 4):
        lg = rng.normal(0, 3, (c, 8, 8)).astype(np.float32)
        p = np.exp(lg) / np.exp(lg).sum(axis=0)
        np.testing.assert_allclose(cb.sigmoid(cb.change_logit(lg)), 1 - p[0], atol=1e-5)


@pytest.mark.parametrize("shape", [(64, 64, 2), (1, 64, 64, 2), (2, 64, 64), (1, 2, 64, 64), (8192,), (2, 4096),
                                   (4096, 2), (1, 1, 4096, 2), (1, 2, 4096), (1, 1, 2, 4096), (1, 4096, 2)])
def test_to_chw_handles_every_layout(shape):
    chw = np.arange(2 * 64 * 64, dtype=np.float32).reshape(2, 64, 64)
    if shape in ((2, 64, 64), (1, 2, 64, 64), (2, 4096), (1, 2, 4096), (1, 1, 2, 4096)):   # planar
        y = chw.reshape(shape)
    else:
        y = chw.transpose(1, 2, 0).reshape(shape)       # channels last, as pyneat hands NCHW back
    assert np.array_equal(cb.to_chw(y, 256), chw)


def test_to_chw_rejects_a_wrong_size():
    with pytest.raises(CywsError, match="values"):
        cb.to_chw(np.zeros((32, 32, 2)), 256)
    for shape in ((128, 64), (2, 32, 128), (4, 2048)):                 # no guessing: unknown layouts raise
        with pytest.raises(CywsError, match="not a layout"):
            cb.to_chw(np.zeros(shape), 256)


@pytest.mark.parametrize("spec,size", [
    (types.SimpleNamespace(shape=[1, 256, 256, 3]), 256),
    (types.SimpleNamespace(shape=(3, 256, 256)), 256),                  # CHW
    (types.SimpleNamespace(shape=lambda: [256, 256, 3]), 256),          # shape is a method
    (types.SimpleNamespace(dims=[1, 128, 128, 3]), 128),                # dims instead of shape
    (types.SimpleNamespace(), 256),                                     # says nothing: the default
])
def test_input_size_reads_any_spec(spec, size):
    assert cb.input_size(spec) == size


def test_input_size_rejects_a_non_square_spec():
    for dims in ([256, 128, 3], [1, 250, 250, 3]):
        with pytest.raises(CywsError, match="square"):
            cb.input_size(types.SimpleNamespace(shape=dims))


def test_fake_backend_is_deterministic_and_checks_inputs():
    be = cb.FakeC3POBackend()
    x = smooth_image(256, 256, seed=1).astype(np.float32) / 255
    y = x.copy()
    y[100:160, 100:160] = (1.0, 0.0, 0.0)
    lg = be.infer(x, y)
    assert lg.shape == (2, 64, 64) and lg.dtype == np.float32
    assert np.array_equal(lg, be.infer(x, y))
    p = be.probability(x, y)
    assert p[32, 32] > 0.9 and p[5, 5] < 0.1 and be.probability(x, x).max() < 0.1
    assert be.stats()["inferences"] == 4 and be.stats()["latency"]["mean_ms"] >= 0
    with pytest.raises(CywsError, match="before"):
        be.infer(np.zeros((128, 128, 3)), y)
    be.close()
    with pytest.raises(CywsError, match="closed"):
        be.infer(x, y)


class FakePyneat:
    """A stand-in for pyneat: two HWC inputs in, the logits back channels last (64, 64, 2)
    from one reused buffer, and every call checked against the accelerator lock."""

    def __init__(self, n_inputs=2, size=256, layout="hwc"):
        self.calls, self.violations, self.pushed, self.max_queued = [], [], [], 0
        self.n_inputs, self.size, self.layout = n_inputs, size, layout
        self.module = self._make()

    def record(self, what):
        self.calls.append(what)
        if not common.ACCELERATOR_LOCK._is_owned():
            self.violations.append(what)

    def _make(self):
        fp = self
        mod = types.ModuleType("pyneat")
        mod.InputKind = types.SimpleNamespace(Tensor="Tensor")
        mod.TensorLayout = types.SimpleNamespace(HWC="HWC")

        class ModelOptions:
            def __init__(self):
                self.preprocess = types.SimpleNamespace(kind=None)

        class Tensor:
            def __init__(self, arr):
                self.arr, self.layout = arr, None

            @classmethod
            def from_numpy(cls, arr, copy=False):
                assert copy is True
                return cls(np.array(arr, copy=True))

            def to_numpy(self):
                return self.arr

        class Run:
            def __init__(self):
                self.queue, self.out = [], np.zeros((64, 64, 2), np.float32)

            def push_tensors(self, ts):
                fp.record("push")
                assert len(ts) == 2 and all(t.layout == "HWC" and t.arr.shape == (fp.size, fp.size, 3) for t in ts)
                fp.pushed.append([t.arr for t in ts])
                self.queue.append([t.arr for t in ts])
                fp.max_queued = max(fp.max_queued, len(self.queue))

            def pull(self, timeout):
                fp.record("pull")
                if not self.queue:
                    return None
                b, a = self.queue.pop(0)
                d = np.abs(a - b).mean(axis=2)[::4, ::4]
                self.out[..., 0], self.out[..., 1] = 0.0, 40 * (d - 0.08)
                out = self.out if fp.layout == "hwc" else self.out.transpose(2, 0, 1)[None]
                return types.SimpleNamespace(tensors=[Tensor(out)])      # a view of the reused buffer

            def close(self):
                fp.record("close")

        class Model:
            def __init__(self, path, opt):
                fp.record("model")
                assert opt.preprocess.kind == "Tensor"

            def input_specs(self):
                return [types.SimpleNamespace(shape=[1, fp.size, fp.size, 3]) for _ in range(fp.n_inputs)]

            def build(self, ts):
                fp.record("build")
                assert len(ts) == 2
                return Run()

        mod.ModelOptions, mod.Tensor, mod.Model = ModelOptions, Tensor, Model
        return mod


@pytest.mark.parametrize("layout", ["hwc", "nchw"])
def test_neat_backend_with_mocked_pyneat(monkeypatch, tmp_path, layout):
    fp = FakePyneat(layout=layout)
    monkeypatch.setitem(sys.modules, "pyneat", fp.module)
    mpk = tmp_path / "c3po_pcd_256_bf16_mpk.tar.gz"
    mpk.write_bytes(b"x")
    be = families.get("c3po").open(mpk, "neat")
    assert isinstance(be, cb.NeatC3POBackend) and be.size == 256
    families.get("c3po").warm(be)
    x = smooth_image(256, 256, seed=2).astype(np.float32) / 255
    y = x.copy()
    y[:64, :64] = 1.0
    a = be.infer(x, y)
    b = be.infer(x, x)
    assert a.shape == (2, 64, 64) and a[1, 5, 5] > 5 and b[1].max() < 0      # copied out: b did not overwrite a
    assert np.array_equal(fp.pushed[-2][0], x) and np.array_equal(fp.pushed[-2][1], y)   # before, then after
    pairs = [(x, y), (x, x), (y, x), (x, y), (y, y)]
    fp.calls.clear()
    outs = be.infer_many(pairs)                                      # pipelined: two pairs in flight
    assert fp.max_queued == 2 and fp.calls[:3] == ["push", "push", "pull"]
    assert fp.calls.count("push") == fp.calls.count("pull") == 5
    for o, (b0, a0) in zip(outs, pairs):                             # in order, each its own copy
        np.testing.assert_array_equal(o, be.infer(b0, a0))
    assert be.stats()["inferences"] == 1 + 2 + 5 + 5
    be.close()
    be.close()
    assert fp.calls.count("close") == 1 and not fp.violations


def test_neat_backend_refuses_a_one_input_model(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "pyneat", FakePyneat(n_inputs=1).module)
    mpk = tmp_path / "dinov2.tar.gz"
    mpk.write_bytes(b"x")
    with pytest.raises(CywsError, match="takes two"):
        cb.NeatC3POBackend(mpk)
    with pytest.raises(CywsError, match="not found"):
        cb.NeatC3POBackend(tmp_path / "missing.tar.gz")
    monkeypatch.setitem(sys.modules, "pyneat", None)
    with pytest.raises(CywsError, match="pyneat is not importable"):
        cb.NeatC3POBackend(mpk)


ONNX = families.C3POFamily.onnx_path("c3po_pcd_256_bf16_mpk.tar.gz")


@pytest.mark.skipif(not ONNX.is_file(), reason="C-3PO ONNX not built (MLSandbox/C3PO build/onnx)")
def test_onnx_backend_matches_its_contract():
    pytest.importorskip("onnxruntime")
    be = cb.OnnxC3POBackend(ONNX)
    x = smooth_image(256, 256, seed=4).astype(np.float32) / 255
    lg = be.infer(x, x)
    assert lg.shape == (2, 64, 64) and np.isfinite(lg).all()
    assert be.probability(x, x).mean() < 0.5                          # the same photo twice: mostly no change


# ---------------------------------------------------------------- app
def _run(before, after, **cfg):
    app = c3po_app.C3POApp(cb.FakeC3POBackend(), c3po_app.C3POConfig(**cfg))
    return app.process(Pair("p", before, after, meta={"load_ms": 2.0}))


def test_app_finds_a_pasted_object():
    before, after = pasted_pair()
    res = _run(before, after)
    assert res.detections, "the pasted box was not found"
    d = res.detections[0]
    assert d.label == "changed" and d.branch == "after" and 0.2 <= d.score <= 1.0
    assert d.x1 < 400 and d.x2 > 260 and d.y1 < 300 and d.y2 > 180
    assert res.mask.dtype == bool and res.mask.shape == (480, 640) and res.mask[240, 330]
    assert res.heat.shape == (120, 160) and res.heat.min() >= 0
    assert res.heat.max() == pytest.approx(c3po_app.HEAT_AT_THR / 0.20, rel=0.1)   # probability ~1 at the box
    assert res.windows[0].level == 0 and (res.windows[0].w, res.windows[0].h) == (640, 480)
    assert d.window_id in {w.id for w in res.windows}
    assert set(res.timings_ms) == set(c3po_app.TIMING_KEYS) and res.timings_ms["load"] == 2.0
    assert res.meta["counts"] == {"appeared": 0, "disappeared": 0, "changed": len(res.detections)}
    assert res.meta["windows_run"] == len(res.windows) and res.meta["backend"] == "fake"
    assert res.registration is None and res.before_aligned is None


def test_app_identical_photos_give_no_change():
    img = smooth_image(640, 480, seed=5)
    for levels in ((0,), (0, 1), (0, 1, 2)):
        res = _run(img, img.copy(), levels=levels, tiles="full")
        assert res.detections == [] and not res.mask.any()


def test_app_levels_gating_and_min_area():
    before, after = pasted_pair()
    assert len(_run(before, after, levels=(0,)).windows) == 1
    full = _run(before, after, levels=(0, 1), tiles="full")
    corner = before.copy()
    corner[20:80, 20:100] = (240, 30, 30)                   # only the top-left tile sees it
    gated = _run(before, corner, levels=(0, 1), tiles="gated")
    fine = _run(before, after, levels=(0, 1, 2), tiles="full")
    assert {w.level for w in full.windows} == {0, 1} and len(full.windows) == 7          # 6 tiles of 0.6
    assert len(gated.windows) == 2 and gated.windows[1].x0 == gated.windows[1].y0 == 0
    assert gated.meta["gated_out"] == 5 and gated.detections
    assert {w.level for w in fine.windows} == {0, 1, 2}
    assert all(w.w == w.h and w.x0 + w.w <= 640 and w.y0 + w.h <= 480 for w in fine.windows[1:])
    tiny = before.copy()
    tiny[10:16, 10:16] = (255, 0, 0)                        # 36 px, far under min_area
    assert _run(before, tiny, levels=(0,)).detections == []
    with pytest.raises(CywsError):
        c3po_app.C3POApp(None, c3po_app.C3POConfig(levels=(1,)))
    with pytest.raises(CywsError):
        c3po_app.C3POApp(None, c3po_app.C3POConfig(thr=1.0))


def test_app_with_registration_returns_the_aligned_photo():
    before, after = pasted_pair()
    res = _run(before, after, register=True)
    assert res.registration is not None and res.before_aligned is not None
    assert res.before_aligned.shape == after.shape and res.detections


# ---------------------------------------------------------------- server
def test_models_table_and_status(demo):
    for name, title, file in (("c3po-pcd", "C-3PO PCD", "c3po_pcd_256_bf16_mpk.tar.gz"),
                              ("c3po-csb", "C-3PO ChangeSim", "c3po_csb_256_bf16_mpk.tar.gz")):
        m = server.MODELS[name]
        assert (m["family"], m["title"], m["file"]) == ("c3po", title, file)
    assert server.MODELS["c3po-pcd"]["label"] == "C-3PO PCD · Street-Trained"
    assert server.MODELS["c3po-csb"]["label"] == "C-3PO ChangeSim · Indoor-Trained"
    assert demo.family.name == "c3po" and isinstance(demo.backend, cb.FakeC3POBackend)
    st = demo.status()
    assert st["family"] == "c3po" and st["capabilities"] == {"per": "pair", "pixel_masks": False, "labels": False}
    assert st["presets"] == families.get("c3po").presets() and st["defaults"]["thr"] == 0.20
    assert st["families"]["c3po"]["title"] == "C-3PO"
    rows = {m["name"]: m for m in st["models"]}
    assert rows["c3po-csb"] == {"name": "c3po-csb", "label": "C-3PO ChangeSim · Indoor-Trained",
                                "title": "C-3PO ChangeSim", "family": "c3po", "available": True, "loaded": False}
    assert rows["c3po-pcd"]["loaded"] is True
    demo.load_model("c3po-csb")
    assert demo.model_name == "c3po-csb" and demo.family.name == "c3po"
    demo.load_model("dinov2-448")
    assert demo.status()["capabilities"]["labels"] is True


def test_c3po_run_through_the_api(demo, tmp_path):
    c = server.create_app(demo).test_client()
    uid = upload(c, tmp_path, *pasted_pair())
    for preset in ("fast", "balanced", "accurate"):
        r = c.post("/api/run", json={"id": uid, "options": {"preset": preset, "mask_mode": "diff"}})
        assert r.status_code == 200, r.get_json()
        out = r.get_json()
        assert out["family"] == "c3po" and out["model"] == "c3po-pcd" and out["config"]["thr"] == 0.20
        assert out["detections"] and {d["label"] for d in out["detections"]} == {"changed"}
        assert out["counts"]["appeared"] == 0 and out["counts"]["disappeared"] == 0
        assert set(out["timings_ms"]) == set(c3po_app.TIMING_KEYS)
    assert out["config"]["levels"] == [0, 1] and out["config"]["tiles"] == "full"
    for k in ("after", "before", "heat", "mask"):
        assert c.get(out["images"][k]).status_code == 200
    bad = c.post("/api/run", json={"id": uid, "options": {"thr": 0.95}})
    assert bad.status_code == 400 and bad.get_json()["error"][0].isupper()
    same = upload(c, tmp_path, pasted_pair()[0], pasted_pair()[0], tag="same")
    assert c.post("/api/run", json={"id": same, "options": {}}).get_json()["detections"] == []
    assert demo.status()["latency"]["mean_ms"] >= 0


def test_hub_accepts_c3po_entries():
    f = [{"name": "c3po_pcd_256_bf16_mpk.tar.gz", "bytes": 10, "sha256": "0" * 64}]
    ok, why = server.Hub.check_entry("c3po-pcd", {"path": "c3po-pcd", "files": f, "input": {"size": 256, "images": 2}}, "c3po")
    assert ok is not None and ok["family"] == "c3po", why
    for inp in ({"size": 448, "images": 2}, {"size": 256, "images": 1}):
        bad, why = server.Hub.check_entry("c3po-pcd", {"path": "c3po-pcd", "files": f, "input": inp}, "c3po")
        assert bad is None and why
    assert server.HUB_REPOS == ["dotimothy/sima-cyws", "dotimothy/sima-dinov2"]     # sima-c3po is not published


def test_cli_accepts_the_c3po_models(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["server.py", "--model", "c3po-csb", "--delete", "nothing",
                                      "--hub-repo", "", "--models", "/nonexistent-dir-for-test", "--port", "1"])
    assert server.main() == 1


def test_publish_hub_dry_run_for_c3po(tmp_path):
    sys.path.insert(0, str(ROOT / "tools"))
    import publish_hub
    fam = publish_hub.FAMILIES["c3po"]
    assert fam["repo"] == "dotimothy/sima-c3po" and set(fam["models"]) == {"c3po-pcd", "c3po-csb"}
    for name, spec in fam["models"].items():
        assert spec["input"] == {"size": 256, "images": 2} and spec["latency_ms"] == 46.7
    r = subprocess.run([sys.executable, str(ROOT / "tools/publish_hub.py"), "--family", "c3po", "--dry-run", str(tmp_path)],
                       capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert r.returncode == 0, r.stderr
    doc = json.loads((tmp_path / "sima-c3po/models.json").read_text())
    assert doc["family"] == "c3po" and doc["version"] == 2
    card = (tmp_path / "sima-c3po/README.md").read_text()
    assert "license: mit" in card and "{rows}" not in card
    for name, e in doc["models"].items():                     # whatever builds are here pass the app's checks
        entry, why = server.Hub.check_entry(name, e, doc["family"])
        assert entry is not None, why


def test_a_missing_c3po_model_says_copy_not_download(tmp_path):
    d = make_demo(tmp_path, files={"bf16"})
    server._safe_load(d, "c3po-csb")
    st = d.status()
    assert st["state"] == "idle"
    assert st["error"] == (f"C-3PO ChangeSim is not on this board yet. Copy c3po_csb_256_bf16_mpk.tar.gz "
                           f"into {d.models_dir}.")
    assert "Download" not in st["error"]
    d.close()
