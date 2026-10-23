"""Model families in the web demo: DINOv2 zero-shot next to CYWS, unload/idle, status, multi-repo hub."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("flask")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "webapp"))
import families  # noqa: E402
import feature_backends  # noqa: E402
import server  # noqa: E402
import zeroshot  # noqa: E402
from common import CywsError, Pair  # noqa: E402
from test_speed import save, smooth_image  # noqa: E402


def make_demo(tmp_path, model="bf16", files=None, hub_repo=None):
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    for n, m in server.MODELS.items():
        if files is None or n in files:
            (models / m["file"]).write_bytes(b"")          # the fake backends never read it
    args = argparse.Namespace(
        port=0, host="127.0.0.1", model=model, models=str(models), data=str(tmp_path / "no-data"),
        cache=str(tmp_path / "cache"), uploads=str(tmp_path / "uploads"), backend="fake", onnx="",
        https=False, tls=False, cert="", key="")
    if hub_repo is not None:
        args.hub_repo = hub_repo
    return server.Demo(args)


@pytest.fixture()
def demo(tmp_path):
    d = make_demo(tmp_path)
    d.load_model("bf16")
    yield d
    d.close()


def pasted_pair(w=640, h=480, seed=3):
    before = smooth_image(w, h, seed=seed)
    after = before.copy()
    after[180:300, 260:400] = (240, 30, 30)                 # a red box appears
    return before, after


def upload(client, tmp_path, before, after, tag=""):
    b = save(tmp_path / f"b{tag}.png", before).read_bytes()
    a = save(tmp_path / f"a{tag}.png", after).read_bytes()
    r = client.post("/api/upload", data={"before": (io.BytesIO(b), "b.png"), "after": (io.BytesIO(a), "a.png")})
    assert r.status_code == 200, r.get_json()
    return r.get_json()["id"]


# ---------------------------------------------------------------- families and the fake feature backend
def test_family_interface_and_presets():
    cy, dn = families.get("cyws"), families.get("dinov2")
    assert families.get("") is cy and families.get(None) is cy
    with pytest.raises(CywsError):
        families.get("yolo")
    assert set(dn.presets()) == {"fast", "balanced", "accurate"}
    assert dn.presets()["fast"] == {"levels": [0], "thr": 0.30}
    assert dn.presets()["accurate"]["tiles"] == "full" and dn.presets()["accurate"]["thr"] == 0.35
    assert dn.defaults() == {"thr": 0.40, "levels": [0, 1], "register": False, "tiles": "gated",
                             "mask_mode": "box", "verify_change": None}
    cfg = dn.config_from(dn.base_config(), {"preset": "fast"})
    assert cfg.levels == (0,) and cfg.thr == 0.30
    cfg = dn.config_from(dn.base_config(), {"preset": "accurate", "levels": "0,1,2", "mask_mode": "diff"})
    assert cfg.levels == (0, 1, 2) and cfg.tiles == "full" and cfg.thr == 0.35
    assert (cfg.radius, cfg.border, cfg.blur, cfg.min_area, cfg.work_long) == (0, 1, 1.0, 0.002, 1024)
    with pytest.raises(CywsError, match="thr"):
        dn.config_from(dn.base_config(), {"thr": 0.05})              # fine for CYWS, not for DINOv2
    cy.config_from(cy.base_config(), {"thr": 0.05})
    spec = dn.describe()["option_spec"]
    assert spec["thr"] == {"type": "float", "range": [0.10, 0.90]}
    assert spec["levels"]["choices"] == ["0", "0,1", "0,1,2"] and spec["mask_mode"]["ignored"] is True
    json.dumps({n: f.describe() for n, f in families.FAMILIES.items()})
    assert cy.capabilities == {"per": "window", "pixel_masks": True, "labels": True}
    assert dn.capabilities == {"per": "image", "pixel_masks": False, "labels": True}


def test_fake_features_are_deterministic_and_shaped():
    be = feature_backends.FakeFeatureBackend()
    x = smooth_image(448, 448, seed=1).astype(np.float32) / 255
    f = be.features(x)
    assert f.shape == (5 + 32 * 32, 384) and f.dtype == np.float32
    assert np.array_equal(f, be.features(x))
    cls, patches = feature_backends.split(f)
    assert cls.shape == (384,) and patches.shape == (32, 32, 384)
    assert be.stats()["latency"]["mean_ms"] >= 0 and be.stats()["inferences"] == 2
    with pytest.raises(CywsError):
        be.features(np.zeros((224, 224, 3), np.float32))
    be.close()
    with pytest.raises(CywsError, match="closed"):
        be.features(x)


def _run_zeroshot(before, after, **cfg):
    be = feature_backends.FakeFeatureBackend()
    app = zeroshot.ZeroShotApp(be, zeroshot.ZeroShotConfig(**cfg))
    return app.process(Pair("p", before, after, meta={"load_ms": 1.5}))


def test_zeroshot_finds_a_pasted_object_with_a_label():
    before, after = pasted_pair()
    res = _run_zeroshot(before, after)
    assert res.detections, "the pasted box was not found"
    d = res.detections[0]
    assert d.label in ("appeared", "disappeared", "changed") and d.branch == "after"
    assert d.x1 < 400 and d.x2 > 260 and d.y1 < 300 and d.y2 > 180          # overlaps the pasted box
    assert res.mask.dtype == bool and res.mask.shape == (480, 640) and res.mask[240, 330]
    assert res.heat.shape == (120, 160) and res.heat.max() > 0.3 and res.heat.min() >= 0
    assert res.windows[0].level == 0 and (res.windows[0].w, res.windows[0].h) == (640, 480)
    assert d.window_id in {w.id for w in res.windows} and d.level in (0, 1)
    assert set(res.timings_ms) == set(zeroshot.TIMING_KEYS) and res.timings_ms["load"] == 1.5
    assert res.meta["windows_run"] == len(res.windows)
    assert sum(res.meta["counts"].values()) == len(res.detections)
    assert res.registration is None and res.before_aligned is None


def test_zeroshot_identical_photos_give_no_change():
    img = smooth_image(640, 480, seed=5)
    for levels in ((0,), (0, 1), (0, 1, 2)):
        res = _run_zeroshot(img, img.copy(), levels=levels, tiles="full")
        assert res.detections == [] and not res.mask.any()
        assert res.meta["counts"] == {"appeared": 0, "disappeared": 0, "changed": 0}


def test_zeroshot_levels_and_gating_choose_the_windows():
    before, after = pasted_pair()
    whole = _run_zeroshot(before, after, levels=(0,))
    assert len(whole.windows) == 1
    full = _run_zeroshot(before, after, levels=(0, 1), tiles="full")
    gated = _run_zeroshot(before, after, levels=(0, 1), tiles="gated")
    fine = _run_zeroshot(before, after, levels=(0, 1, 2), tiles="full")
    assert len(full.windows) > 1 and {w.level for w in full.windows} == {0, 1}
    assert 1 <= len(gated.windows) <= len(full.windows)
    assert gated.meta["gated_out"] == len(full.windows) - len(gated.windows)
    assert {w.level for w in fine.windows} == {0, 1, 2}
    assert all(w.x0 + w.w <= 640 and w.y0 + w.h <= 480 for w in fine.windows)
    t = full.windows[1]
    assert t.img_edges[0] == (t.x0 == 0) and t.img_edges[2] == (t.x0 + t.w >= 640)
    with pytest.raises(CywsError):
        zeroshot.ZeroShotApp(None, zeroshot.ZeroShotConfig(levels=(1,)))


def test_zeroshot_with_registration_returns_the_aligned_photo():
    before, after = pasted_pair()
    res = _run_zeroshot(before, after, register=True)
    assert res.registration is not None and res.before_aligned is not None
    assert res.before_aligned.shape == after.shape and res.detections


# ---------------------------------------------------------------- the demo: dispatch, unload, status
def test_load_dispatches_on_the_family(demo):
    assert demo.family.name == "cyws" and type(demo.backend).__name__ == "FakeBackend"
    demo.load_model("dinov2-448")
    assert demo.family.name == "dinov2" and isinstance(demo.backend, feature_backends.FakeFeatureBackend)
    st = demo.status()
    assert st["family"] == "dinov2" and st["capabilities"]["per"] == "image"
    assert st["presets"] == families.get("dinov2").presets() and st["defaults"]["thr"] == 0.40
    demo.load_model("w8a16")
    assert demo.family.name == "cyws" and demo.status()["defaults"]["thr"] == pytest.approx(0.29)


def test_dinov2_run_through_the_api(demo, tmp_path):
    c = server.create_app(demo).test_client()
    assert c.post("/api/models/load", json={"name": "dinov2-448"}).get_json()["family"] == "dinov2"
    uid = upload(c, tmp_path, *pasted_pair())
    r = c.post("/api/run", json={"id": uid, "options": {"preset": "balanced", "mask_mode": "diff"}})
    assert r.status_code == 200, r.get_json()
    out = r.get_json()
    assert out["family"] == "dinov2" and out["model"] == "dinov2-448"
    assert out["detections"] and out["detections"][0]["label"] in ("appeared", "disappeared", "changed")
    assert out["detections"][0]["branch"] == "after"
    assert out["windows"][0] == {"id": 0, "level": 0, "x0": 0, "y0": 0, "w": out["size"]["w"], "h": out["size"]["h"]}
    assert set(out["timings_ms"]) == {"load", "register", "stream", "postprocess", "total"}
    assert out["config"]["thr"] == 0.40 and out["config"]["levels"] == [0, 1]
    for k in ("after", "before", "heat", "mask"):
        assert c.get(out["images"][k]).status_code == 200
    bad = c.post("/api/run", json={"id": uid, "options": {"thr": 0.95}})
    assert bad.status_code == 400 and bad.get_json()["error"][0].isupper()
    same = upload(c, tmp_path, pasted_pair()[0], pasted_pair()[0], tag="same")
    assert c.post("/api/run", json={"id": same, "options": {}}).get_json()["detections"] == []
    assert demo.status()["latency"]["mean_ms"] >= 0


def test_unload_goes_idle_and_runs_are_refused(demo, tmp_path):
    c = server.create_app(demo).test_client()
    uid = upload(c, tmp_path, *pasted_pair())
    st = c.post("/api/models/unload").get_json()
    assert st["state"] == "idle" and st["model"] is None and st["family"] is None
    assert demo.backend is None and st["capabilities"] == families.get("cyws").capabilities
    r = c.post("/api/run", json={"id": uid, "options": {}})
    assert r.status_code == 409 and r.get_json()["error"] == "No model is loaded. Load one in Models."
    assert c.post("/api/models/unload").get_json()["state"] == "idle"       # twice is fine
    st = c.post("/api/models/load", json={"name": "dinov2-448"}).get_json()
    assert st["state"] == "ready" and st["model"] == "dinov2-448"
    assert c.post("/api/run", json={"id": uid, "options": {}}).status_code == 200
    assert c.post("/api/hub/delete", json={"name": "dinov2-448"}).status_code == 400   # loaded
    c.post("/api/models/unload")
    assert c.post("/api/hub/delete", json={"name": "dinov2-448"}).status_code == 200   # unloaded: fine


def test_status_fields(demo):
    st = demo.status()
    assert st["state"] == "ready" and st["family"] == "cyws"
    assert st["capabilities"] == {"per": "window", "pixel_masks": True, "labels": True}
    assert set(st["families"]) == {"cyws", "dinov2", "c3po"}
    for f in st["families"].values():
        assert {"title", "presets", "defaults", "option_spec", "capabilities"} <= set(f)
    assert st["families"]["dinov2"]["title"] == "DINOv2"
    rows = {m["name"]: m for m in st["models"]}
    assert set(rows) >= {"bf16", "w8a16", "dinov2-448"}
    assert rows["dinov2-448"] == {"name": "dinov2-448", "label": "DINOv2 448 · Zero-Shot", "title": "DINOv2 448",
                                  "family": "dinov2", "available": True, "loaded": False}
    assert rows["bf16"]["title"] == "CYWS bf16" and rows["bf16"]["loaded"] is True
    json.dumps(st)


def test_startup_with_a_missing_model_is_idle(tmp_path):
    d = make_demo(tmp_path, files={"w8a16"})
    server._safe_load(d, "bf16")
    st = d.status()
    assert st["state"] == "idle" and st["model"] is None
    assert st["error"] == "CYWS bf16 is not on this board yet. Download it in Models."
    d.load_model("w8a16")
    assert d.status()["state"] == "ready"
    with pytest.raises(CywsError, match="not on this board"):
        d.load_model("dinov2-448")
    assert d.status()["state"] == "ready" and d.model_name == "w8a16"      # still loaded
    d.close()


def test_cli_accepts_every_builtin_model(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["server.py", "--model", "dinov2-448", "--delete", "nothing",
                                      "--hub-repo", "", "--models", "/nonexistent-dir-for-test", "--port", "1"])
    assert server.main() == 1                                         # parsed; delete says "no such model"


# ---------------------------------------------------------------- several Hugging Face repositories
def _file(name, blob):
    return [{"name": name, "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}]


class _MultiHF:
    """huggingface.co with one models.json per repository, plus the MPKs."""

    def __init__(self, docs: dict, blobs: dict):
        self.docs, self.blobs, self.urls = docs, blobs, []

    def __call__(self, req, timeout=None):
        class R(io.BytesIO):
            status = 200
            def __enter__(self): return self
            def __exit__(self, *a): self.close()
        url = req.full_url
        self.urls.append(url)
        if "huggingface.co/" not in url:
            raise OSError("connection refused")                    # e.g. the demo's /api/status
        repo = url.split("huggingface.co/")[1].split("/resolve/")[0]
        if url.endswith("models.json"):
            if repo not in self.docs:
                raise OSError("HTTP Error 404: Not Found")
            return R(json.dumps(self.docs[repo]).encode())
        return R(self.blobs[url.rsplit("/", 1)[1]])


@pytest.fixture()
def two_repos(monkeypatch):
    cy_blob, dn_blob = b"cyws" * 2000, b"dino" * 3000
    docs = {
        "a/sima-cyws": {"models": {                                 # version 1, no family: cyws
            "tiny": {"title": "CYWS tiny", "path": "tiny", "files": _file("cyws_tiny_mpk.tar.gz", cy_blob)},
            "clash": {"title": "First", "path": "clash", "files": _file("cyws_clash_mpk.tar.gz", cy_blob)},
        }},
        "b/sima-dinov2": {"version": 2, "family": "dinov2", "models": {
            "dinov2-448": {"title": "DINOv2 448", "path": "dinov2-448", "input": {"size": 448, "images": 1},
                           "card": {"license": "Apache-2.0", "upstream": "facebookresearch/dinov2"},
                           "files": _file("dinov2_vits14_reg_448_bf16_mpk.tar.gz", dn_blob)},
            "odd": {"title": "Bad size", "path": "odd", "input": {"size": 450, "images": 1},
                    "files": _file("dinov2_odd_mpk.tar.gz", dn_blob)},
            "alien": {"title": "Unknown", "path": "alien", "family": "yolo",
                      "files": _file("alien_mpk.tar.gz", dn_blob)},
            "broken": {"title": "No sum", "path": "broken", "files": [{"name": "x.tar.gz", "bytes": 1}]},
            "clash": {"title": "Second", "path": "clash", "files": _file("dinov2_clash_mpk.tar.gz", dn_blob)},
        }},
    }
    fake = _MultiHF(docs, {"cyws_tiny_mpk.tar.gz": cy_blob, "cyws_clash_mpk.tar.gz": cy_blob,
                           "dinov2_vits14_reg_448_bf16_mpk.tar.gz": dn_blob})
    monkeypatch.setattr(server.urllib.request, "urlopen", fake)
    return fake


def test_parse_repos():
    assert server.parse_repos("") == [] and server.parse_repos(None) == []
    assert server.parse_repos("a/x, b/y,,a/x") == ["a/x", "b/y"]
    assert server.parse_repos(["a/x", "b/y"]) == ["a/x", "b/y"]
    assert server.HUB_REPOS == ["dotimothy/sima-cyws", "dotimothy/sima-dinov2"]


def test_hub_reads_every_repository(tmp_path, two_repos):
    hub = server.Hub("a/sima-cyws,b/sima-dinov2", tmp_path, loaded=lambda: None, catalog=lambda: {})
    hub._refresh()
    d = hub.describe()
    assert d["repo"] == "a/sima-cyws" and d["listed"] is True and d["error"] is None
    assert [r["repo"] for r in d["repos"]] == ["a/sima-cyws", "b/sima-dinov2"]
    assert all(r["listed"] and r["error"] is None for r in d["repos"])
    assert set(d["models"]) == {"tiny", "clash", "dinov2-448"}
    tiny, dino = d["models"]["tiny"], d["models"]["dinov2-448"]
    assert tiny["family"] == "cyws" and tiny["repo"] == "a/sima-cyws" and tiny["input"] is None
    assert dino["family"] == "dinov2" and dino["repo"] == "b/sima-dinov2"
    assert dino["input"] == {"size": 448, "images": 1} and dino["card"]["license"] == "Apache-2.0"
    assert dino["url"] == ("https://huggingface.co/b/sima-dinov2/resolve/main/dinov2-448/"
                           "dinov2_vits14_reg_448_bf16_mpk.tar.gz")
    assert d["models"]["clash"]["title"] == "First" and d["models"]["clash"]["repo"] == "a/sima-cyws"
    skipped = {(s["repo"], s["name"]) for s in d["skipped"]}
    assert skipped == {("b/sima-dinov2", n) for n in ("odd", "alien", "broken", "clash")}
    assert all(s["reason"] for s in d["skipped"])


def test_hub_download_uses_the_entry_repo(tmp_path, two_repos):
    hub = server.Hub(["a/sima-cyws", "b/sima-dinov2"], tmp_path, loaded=lambda: None, catalog=lambda: {})
    hub._refresh()
    hub.download("dinov2-448")
    hub._job["thread"].join(10)
    assert hub.describe()["models"]["dinov2-448"]["state"] == "on_board"
    assert any("b/sima-dinov2/resolve/main/dinov2-448/" in u for u in two_repos.urls)
    assert hub.verify("dinov2-448") is True


def test_one_missing_repository_does_not_hide_the_other(tmp_path, two_repos):
    hub = server.Hub("a/sima-cyws,c/missing", tmp_path, loaded=lambda: None, catalog=lambda: {})
    hub._refresh()
    d = hub.describe()
    assert set(d["models"]) == {"tiny", "clash"} and d["listed"] is False   # spec: listed = all repos listed
    assert d["error"] and "c/missing" in d["error"]
    assert d["repos"][1] == {"repo": "c/missing", "page": "https://huggingface.co/c/missing", "listed": False,
                             "error": d["error"]}
    assert d["collection"] is None


def test_hub_models_join_the_catalog_with_their_family(tmp_path, two_repos):
    d = make_demo(tmp_path, hub_repo="a/sima-cyws,b/sima-dinov2")
    d.hub.models(wait=True)
    rows = {m["name"]: m for m in d.available_models()}
    assert rows["tiny"]["family"] == "cyws" and rows["tiny"]["title"] == "CYWS tiny"
    assert rows["dinov2-448"]["family"] == "dinov2" and rows["dinov2-448"]["label"] == "DINOv2 448 · Zero-Shot"
    assert rows["dinov2-448"]["available"] is False             # an empty file is not the published size
    d.close()


def test_cli_over_two_repositories(tmp_path, two_repos, capsys):
    m = tmp_path / "m"
    args = argparse.Namespace(hub_repo="a/sima-cyws,b/sima-dinov2", models=str(m), list_hub=True, fetch=None,
                              verify=None, delete=None, port=1)
    assert server.hub_command(args) == 0
    out = capsys.readouterr()
    assert "a/sima-cyws" in out.out and "b/sima-dinov2" in out.out
    assert "dinov2-448   DINOv2 448" in out.out and "dinov2" in out.out
    assert "skipped odd" in out.err
    args.list_hub, args.fetch = False, ["tiny", "dinov2-448"]
    assert server.hub_command(args) == 0
    assert (m / "dinov2_vits14_reg_448_bf16_mpk.tar.gz").exists() and (m / "cyws_tiny_mpk.tar.gz").exists()
    args.fetch, args.verify = None, ["all"]
    assert server.hub_command(args) == 0
    args.verify, args.fetch = None, ["nope"]
    assert server.hub_command(args) == 1
    assert "a/sima-cyws, b/sima-dinov2" in capsys.readouterr().err
    args.fetch, args.delete = None, ["dinov2-448", "tiny"]
    assert server.delete_command(args) == 0
    assert not list(m.glob("*_mpk.tar.gz"))
    args.hub_repo = "c/missing"
    args.delete, args.list_hub = None, True
    assert server.hub_command(args) == 1


# ---------------------------------------------------------------- review fixes
def test_a_non_numeric_option_is_a_clean_400(demo, tmp_path):
    c = server.create_app(demo).test_client()
    uid = upload(c, tmp_path, *pasted_pair())
    for name in ("bf16", "dinov2-448"):
        c.post("/api/models/load", json={"name": name})
        r = c.post("/api/run", json={"id": uid, "options": {"thr": "abc"}})
        assert r.status_code == 400 and r.get_json()["error"] == "Thr must be a number"


def test_cli_delete_after_unload(tmp_path, monkeypatch, capsys):
    d = make_demo(tmp_path)
    d.load_model("dinov2-448")
    d.unload_model()
    st = d.status()
    assert st["model"] is None and st["load"]["target"] == "dinov2-448"     # the last load, kept for timing

    class Reply(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        return Reply(json.dumps(served).encode())

    monkeypatch.setattr(server.urllib.request, "urlopen", fake_urlopen)
    args = argparse.Namespace(hub_repo="", models=str(tmp_path / "models"), delete=["dinov2-448"], port=1)
    served = dict(st, load=dict(st["load"], stage="load"), state="loading")    # still loading: refused
    assert server.delete_command(args) == 1
    assert "Unload it in Models" in capsys.readouterr().err
    served = st                                                              # unloaded: deleted
    assert server.delete_command(args) == 0
    assert not (tmp_path / "models" / server.MODELS["dinov2-448"]["file"]).exists()
    d.close()


def test_a_repository_that_errors_backs_off(tmp_path, two_repos, monkeypatch):
    hub = server.Hub("a/sima-cyws,c/missing", tmp_path, loaded=lambda: None, catalog=lambda: {})
    hub._refresh()
    started = []
    monkeypatch.setattr(server.threading, "Thread",
                        lambda **kw: type("T", (), {"start": lambda s: started.append(kw), "join": lambda s, t: 0})())
    clock = [hub._fetched + 60]                                  # well past 15 s, short of REFRESH
    monkeypatch.setattr(server.time, "monotonic", lambda: clock[0])
    hub.models()
    assert not started
    clock[0] = hub._fetched + hub.REFRESH + 1
    hub.models()
    assert len(started) == 1


def test_no_list_at_all_retries_soon(tmp_path, two_repos, monkeypatch):
    hub = server.Hub("c/missing", tmp_path, loaded=lambda: None, catalog=lambda: {})
    hub._refresh()
    assert hub._models is None and hub.describe()["listed"] is False
    started = []
    monkeypatch.setattr(server.threading, "Thread",
                        lambda **kw: type("T", (), {"start": lambda s: started.append(kw), "join": lambda s, t: 0})())
    monkeypatch.setattr(server.time, "monotonic", lambda: hub._fetched + 16)
    hub.models()
    assert len(started) == 1
