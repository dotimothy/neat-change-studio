"""Web demo server: live webcam uploads (before_from), model-switch stages, error text."""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("flask")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "webapp"))
import server  # noqa: E402
from test_speed import save, smooth_image  # noqa: E402


@pytest.fixture()
def demo(tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    for m in server.MODELS.values():
        (models / m["file"]).write_bytes(b"")          # the fake backend never reads it
    args = argparse.Namespace(
        port=0, host="127.0.0.1", model="bf16", models=str(models), data=str(tmp_path / "no-data"),
        cache=str(tmp_path / "cache"), uploads=str(tmp_path / "uploads"), backend="fake", onnx="",
        https=False, tls=False, cert="", key="")
    d = server.Demo(args)
    d.load_model("bf16")
    yield d
    d.close()


@pytest.fixture()
def client(demo):
    return server.create_app(demo).test_client()


def jpeg_bytes(tmp_path, name, seed):
    return save(tmp_path / name, smooth_image(640, 480, seed=seed)).read_bytes()


def test_live_upload_reuses_the_before_photo(client, demo, tmp_path):
    b, a1, a2 = (jpeg_bytes(tmp_path, f"{n}.jpg", i) for i, n in enumerate(("b", "a1", "a2")))
    r1 = client.post("/api/upload", data={"before": (io.BytesIO(b), "b.jpg"), "after": (io.BytesIO(a1), "a1.jpg")})
    assert r1.status_code == 200
    uid1 = r1.get_json()["id"]
    r2 = client.post("/api/upload", data={"before_from": uid1, "after": (io.BytesIO(a2), "a2.jpg")})
    assert r2.status_code == 200, r2.get_json()
    uid2 = r2.get_json()["id"]
    item1, item2 = demo.uploads[uid1], demo.uploads[uid2]
    assert item2.before_path != item1.before_path                  # its own copy: survives eviction of uid1
    assert item2.before_path.read_bytes() == b
    assert item2.after_path.read_bytes() == a2
    run = client.post("/api/run", json={"id": uid2, "options": {}})
    assert run.status_code == 200 and run.get_json()["pair"] == uid2


def test_live_upload_from_an_expired_before(client, tmp_path):
    a = jpeg_bytes(tmp_path, "a.jpg", 1)
    r = client.post("/api/upload", data={"before_from": "upload999", "after": (io.BytesIO(a), "a.jpg")})
    assert r.status_code == 400
    msg = r.get_json()["error"]
    assert "expired" in msg and msg[0].isupper()
    r = client.post("/api/upload", data={"after": (io.BytesIO(a), "a.jpg")})
    assert r.status_code == 400


def test_errors_are_capitalised_and_unquoted(client):
    r = client.post("/api/run", json={"id": "nope", "options": {}})
    assert r.status_code in (400, 404)
    msg = r.get_json()["error"]
    assert msg[0].isupper() and not msg.startswith("'")


def test_model_switch_reports_its_stages(demo, monkeypatch):
    seen = []
    real = server.backends.make_backend

    def spy(*a, **kw):
        seen.append(("build", demo.load_stage, demo.load_target))
        be = real(*a, **kw)
        real_infer = be.infer

        def infer(*x, **k):
            seen.append(("infer", demo.load_stage))
            return real_infer(*x, **k)
        be.infer = infer
        return be

    old_close = demo.backend.close

    def close():
        seen.append(("close", demo.load_stage))
        old_close()
    demo.backend.close = close
    monkeypatch.setattr(server.backends, "make_backend", spy)
    demo.load_model("w8a16")
    assert seen == [("close", "release"), ("build", "load", "w8a16"), ("infer", "warm")]
    st = demo.status()
    assert st["state"] == "ready" and st["model"] == "w8a16"
    assert st["load"]["stage"] == "" and st["load"]["target"] == "w8a16"
    assert st["load"]["expected_s"] is not None and st["load"]["expected_s"] >= 0
    assert st["https_port"] is None


def test_failed_switch_clears_the_stage(demo, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("no MLA")
    monkeypatch.setattr(server.backends, "make_backend", boom)
    with pytest.raises(RuntimeError):
        demo.load_model("w8a16")
    st = demo.status()
    assert st["state"] == "error" and st["load"]["stage"] == ""


def test_one_port_serves_http_and_https(demo, tmp_path):
    import shutil
    import ssl
    import subprocess
    import threading
    import urllib.request
    from werkzeug.serving import make_server

    if not shutil.which("openssl"):
        pytest.skip("openssl not installed")
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-subj", "/CN=test",
                    "-keyout", str(key), "-out", str(cert)], check=True, capture_output=True)
    srv = make_server("127.0.0.1", 0, server.create_app(demo), threaded=True)
    server.serve_both_protocols(srv, str(cert), str(key))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        port = srv.server_port
        plain = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=10).read()
        ctx = ssl.create_default_context()
        ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
        secure = urllib.request.urlopen(f"https://127.0.0.1:{port}/api/status", timeout=10, context=ctx).read()
        assert b"cyws-demo" in plain and b"cyws-demo" in secure
    finally:
        srv.shutdown()


# ---------------------------------------------------------------- model hub (Hugging Face)
class _FakeHF:
    """Stands in for huggingface.co: models.json and one MPK, with Range support."""

    def __init__(self, blob: bytes, sha: str | None = None):
        import hashlib
        self.blob = blob
        self.doc = {"models": {"tiny": {"title": "CYWS tiny", "about": "test", "path": "tiny", "precision": "BF16",
                                        "latency_ms": 1.0, "accuracy": {"changes_found": "1 / 1"},
                                        "files": [{"name": "cyws_tiny_mpk.tar.gz", "bytes": len(blob),
                                                   "sha256": sha or hashlib.sha256(blob).hexdigest()}]}}}
        self.ranges = []

    def __call__(self, req, timeout=None):
        import json as _json

        class R(io.BytesIO):
            status = 200
            def __enter__(self): return self
            def __exit__(self, *a): self.close()
        url = req.full_url
        if url.endswith("models.json"):
            return R(_json.dumps(self.doc).encode())
        rng = req.get_header("Range")
        self.ranges.append(rng)
        if rng:
            r = R(self.blob[int(rng.split("=")[1].rstrip("-")):])
            r.status = 206
            return r
        return R(self.blob)


@pytest.fixture()
def hub_demo(demo, tmp_path, monkeypatch):
    fake = _FakeHF(b"mpk-bytes" * 5000)
    monkeypatch.setattr(server.urllib.request, "urlopen", fake)
    demo.hub = server.Hub("someone/sima-cyws", demo.models_dir, loaded=lambda: demo.model_name, catalog=demo._catalog)
    demo.hub.models(wait=True)
    return demo, fake


def _wait(hub):
    job = hub._job
    if job is not None:
        job["thread"].join(10)


def test_hub_lists_and_downloads_a_checked_model(hub_demo):
    demo, fake = hub_demo
    d = demo.hub.describe()
    assert d["error"] is None and d["models"]["tiny"]["state"] == "available"
    assert "tiny" in [m["name"] for m in demo.available_models() if not m["available"]]
    demo.hub.download("tiny")
    _wait(demo.hub)
    d = demo.hub.describe()["models"]["tiny"]
    assert d["state"] == "on_board" and d["error"] is None
    assert (demo.models_dir / "cyws_tiny_mpk.tar.gz").read_bytes() == fake.blob
    assert {"name": "tiny", "available": True}.items() <= next(m for m in demo.available_models() if m["name"] == "tiny").items()
    with pytest.raises(server.CywsError, match="already"):
        demo.hub.download("tiny")


def test_hub_rejects_a_damaged_download(demo, monkeypatch):
    fake = _FakeHF(b"x" * 1000, sha="0" * 64)
    monkeypatch.setattr(server.urllib.request, "urlopen", fake)
    demo.hub = server.Hub("someone/sima-cyws", demo.models_dir, loaded=lambda: demo.model_name, catalog=demo._catalog)
    demo.hub.models(wait=True)
    demo.hub.download("tiny")
    _wait(demo.hub)
    d = demo.hub.describe()["models"]["tiny"]
    assert d["state"] == "available" and "damaged" in d["error"]
    assert not list(demo.models_dir.glob("cyws_tiny*"))


def test_hub_resumes_a_stopped_download(hub_demo):
    demo, fake = hub_demo
    part = demo.models_dir / "cyws_tiny_mpk.tar.gz.part"
    part.write_bytes(fake.blob[:12345])
    assert demo.hub.describe()["models"]["tiny"]["partial_bytes"] == 12345
    demo.hub.download("tiny")
    _wait(demo.hub)
    assert fake.ranges[-1] == "bytes=12345-"
    assert (demo.models_dir / "cyws_tiny_mpk.tar.gz").read_bytes() == fake.blob


def test_hub_routes(hub_demo):
    demo, _ = hub_demo
    c = server.create_app(demo).test_client()
    assert c.get("/api/hub").get_json()["models"]["tiny"]["state"] == "available"
    r = c.post("/api/hub/download", json={"name": "nope"})
    assert r.status_code == 400 and r.get_json()["error"][0].isupper()
    assert c.post("/api/hub/explode", json={"name": "tiny"}).status_code == 404


def test_fetch_command_downloads_and_skips(tmp_path, monkeypatch, capsys):
    fake = _FakeHF(b"mpk" * 3000)
    monkeypatch.setattr(server.urllib.request, "urlopen", fake)
    args = argparse.Namespace(hub_repo="someone/sima-cyws", models=str(tmp_path / "m"), list_hub=False, fetch=["all"])
    assert server.hub_command(args) == 0
    assert (tmp_path / "m" / "cyws_tiny_mpk.tar.gz").read_bytes() == fake.blob
    assert server.hub_command(args) == 0
    assert "already here" in capsys.readouterr().out
    args.fetch = ["nope"]
    assert server.hub_command(args) == 1
    args.fetch, args.list_hub = None, True
    assert server.hub_command(args) == 0
    listing = capsys.readouterr().out
    assert "tiny" in listing and "on board" in listing


def test_default_models_dir_precedence(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setenv("CYWS_MODELS_DIR", str(tmp_path / "env"))
    assert server.default_models_dir() == tmp_path / "env"
    monkeypatch.delenv("CYWS_MODELS_DIR")
    (tmp_path / ".models-dir").write_text(str(tmp_path / "remembered") + "\n")
    assert server.default_models_dir() == tmp_path / "remembered"
    (tmp_path / ".models-dir").unlink()
    expect = Path("/media/nvme/models/cyws") if Path("/media/nvme").is_dir() else tmp_path / "models"
    assert server.default_models_dir() == expect


def test_free_bytes_of_a_missing_dir(tmp_path):
    assert server._free_bytes(tmp_path / "not" / "yet") > 0


# ---------------------------------------------------------------- model manager
def test_delete_a_downloaded_model(hub_demo):
    demo, fake = hub_demo
    c = server.create_app(demo).test_client()
    demo.hub.download("tiny")
    _wait(demo.hub)
    d = c.get("/api/hub").get_json()
    assert d["model_count"] >= 1 and d["used_bytes"] >= len(fake.blob)
    assert d["models"]["tiny"]["verified"] is True                  # a fresh download was checked
    r = c.post("/api/hub/delete", json={"name": "tiny"})
    assert r.status_code == 200, r.get_json()
    assert r.get_json()["models"]["tiny"]["state"] == "available"
    assert not (demo.models_dir / "cyws_tiny_mpk.tar.gz").exists()
    r = c.post("/api/hub/delete", json={"name": "tiny"})
    assert r.status_code == 400 and "not on this board" in r.get_json()["error"]


def test_the_loaded_model_cannot_be_deleted(hub_demo):
    demo, _ = hub_demo
    c = server.create_app(demo).test_client()
    r = c.post("/api/hub/delete", json={"name": "bf16"})
    assert r.status_code == 400 and "Load another model" in r.get_json()["error"]
    assert (demo.models_dir / server.MODELS["bf16"]["file"]).exists()
    r = c.post("/api/hub/delete", json={"name": "w8a16"})            # not loaded: fine, even without the hub
    assert r.status_code == 200
    assert not (demo.models_dir / server.MODELS["w8a16"]["file"]).exists()
    assert c.post("/api/hub/delete", json={"name": "../etc"}).status_code == 400


def test_delete_also_removes_a_stopped_download(hub_demo):
    demo, fake = hub_demo
    part = demo.models_dir / "cyws_tiny_mpk.tar.gz.part"
    part.write_bytes(fake.blob[:100])
    assert demo.hub.describe()["partial_count"] == 1
    demo.delete_model("tiny")
    assert not part.exists()


def test_verify_and_redownload_a_damaged_model(hub_demo):
    demo, fake = hub_demo
    c = server.create_app(demo).test_client()
    path = demo.models_dir / "cyws_tiny_mpk.tar.gz"
    path.write_bytes(b"Z" * len(fake.blob))                          # right size, wrong bytes
    assert c.get("/api/hub").get_json()["models"]["tiny"]["state"] == "on_board"
    r = c.post("/api/hub/verify", json={"name": "tiny"}).get_json()
    assert r["verified"] == {"name": "tiny", "ok": False}
    assert r["models"]["tiny"]["state"] == "damaged" and "Re-download" in r["models"]["tiny"]["error"]
    assert c.post("/api/hub/download", json={"name": "tiny"}).status_code == 400   # already here: needs redownload
    assert c.post("/api/hub/redownload", json={"name": "tiny"}).status_code == 200
    _wait(demo.hub)
    assert path.read_bytes() == fake.blob
    m = c.get("/api/hub").get_json()["models"]["tiny"]
    assert m["state"] == "on_board" and m["verified"] is True and m["error"] is None
    assert c.post("/api/hub/verify", json={"name": "tiny"}).get_json()["verified"]["ok"] is True


def test_a_failed_redownload_keeps_the_old_file(hub_demo, monkeypatch):
    demo, fake = hub_demo
    path = demo.models_dir / "cyws_tiny_mpk.tar.gz"
    path.write_bytes(fake.blob)

    def offline(req, timeout=None):
        raise OSError("network is down")
    monkeypatch.setattr(server.urllib.request, "urlopen", offline)
    demo.hub.redownload("tiny")
    _wait(demo.hub)
    assert path.read_bytes() == fake.blob
    d = demo.hub.describe()["models"]["tiny"]
    assert d["state"] == "on_board" and "stopped" in d["error"]


def test_a_downloading_model_cannot_be_deleted(hub_demo):
    import threading
    demo, _ = hub_demo
    gate = threading.Event()
    demo.hub._job = {"name": "tiny", "thread": threading.Thread(target=gate.wait), "done": 0, "fetched": 0,
                     "started": 0, "cancel": threading.Event()}
    demo.hub._job["thread"].start()
    try:
        with pytest.raises(server.CywsError, match="downloading"):
            demo.delete_model("tiny")
    finally:
        gate.set()


def test_delete_and_verify_commands(tmp_path, monkeypatch, capsys):
    fake = _FakeHF(b"mpk" * 3000)
    monkeypatch.setattr(server.urllib.request, "urlopen", fake)
    m = tmp_path / "m"
    args = argparse.Namespace(hub_repo="someone/sima-cyws", models=str(m), list_hub=False, fetch=["tiny"],
                              verify=None, delete=None, port=1)
    assert server.hub_command(args) == 0
    args.fetch, args.verify = None, ["all"]
    assert server.hub_command(args) == 0
    assert "matches the published file" in capsys.readouterr().out
    (m / "cyws_tiny_mpk.tar.gz").write_bytes(b"x" * 9000)
    assert server.hub_command(args) == 1
    args.verify, args.delete = None, ["tiny"]
    assert server.delete_command(args) == 0
    assert not (m / "cyws_tiny_mpk.tar.gz").exists()
    assert "deleted" in capsys.readouterr().out
    args.delete = ["nope"]
    assert server.delete_command(args) == 1
