#!/usr/bin/env python3
"""CYWS change-detection web demo on the SiMa Modalix MLA.

Pick a before/after pair from the test set (or upload your own two photos);
CYWS runs on the MLA through pyneat (Model -> Graph -> Run) via the app in
../app, and the page shows the detected changes on the "after" image together
with the heat map, the change mask, the ground truth (when the pair has one),
per-stage timings and accuracy against the ground truth.

    python3 webapp/server.py [--port 8098] [--model bf16] [--data data/testCYWS]

One MLA workload at a time: every pipeline run and every model switch holds one
lock, and the backend itself serialises accelerator calls (common.ACCELERATOR_LOCK).
"""
from __future__ import annotations

import argparse
import dataclasses
import faulthandler
import io
import itertools
import hashlib
import json
import mimetypes
import os
import re
import shutil
import signal
import socket
import ssl
import sys
import threading
import time
import traceback
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.serving import make_server

ROOT = Path(__file__).resolve().parent.parent          # repo root, or /media/nvme/cyws on the board
WEB = Path(__file__).resolve().parent / "web"
sys.path.insert(0, str(ROOT / "app"))
import backends  # noqa: E402
import ingestion  # noqa: E402
import post_process  # noqa: E402
from app import CywsApp  # noqa: E402
from common import AppConfig, CywsError  # noqa: E402
from dataset_loader import discover_pairs, static_pair  # noqa: E402

mimetypes.add_type("font/woff2", ".woff2")      # not in every platform's mime table
mimetypes.add_type("image/svg+xml", ".svg")

VIEW_LONG = 1600          # long side of the images sent to the browser
MAX_RESULTS = 24          # rendered results kept in memory
MAX_DECODED = 6           # decoded pairs kept, so re-runs with new settings skip the JPEG decode
MAX_UPLOADS = 20
MODELS = OrderedDict([
    ("bf16", {"file": "cyws_bf16_mpk.tar.gz", "label": "bf16 · Most Faithful"}),
    ("w8a16", {"file": "cyws_w8a16_mpk.tar.gz", "label": "w8a16 · Faster"}),
])
HUB_REPO = "dotimothy/sima-cyws"   # compiled models on Hugging Face (tools/publish_hub.py)
HUB_COLLECTION = "dotimothy/simaai-change-detection-models-6ada4e701e447e9db528596b"

# Settings the page may change, with their allowed values (anything else is rejected).
OPTION_SPEC = {
    "thr": (float, 0.05, 0.95),
    "levels": (str, ("0", "0,1", "0,1,2")),
    "register": (bool,),
    "tiles": (str, ("full", "gated")),
    "mask_mode": (str, ("box", "diff")),
    "verify_change": (bool,),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- images
def _fit(img: np.ndarray, long: int = VIEW_LONG) -> np.ndarray:
    h, w = img.shape[:2]
    s = long / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def _fit_hw(h: int, w: int, long: int = VIEW_LONG) -> tuple[int, int]:
    """(h, w) that _fit produces for an (h, w) image."""
    s = long / max(h, w)
    return (h, w) if s >= 1 else (round(h * s), round(w * s))


def jpeg(rgb: np.ndarray, quality: int = 86) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def png_rgba(rgba: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def heat_layer(heat: np.ndarray) -> bytes:
    """Heat canvas as a transparent colour map: alpha follows the heat value.

    Sent at the canvas's own resolution (1/4 of the working frame); the browser
    stretches every layer over the photo, and smooth upscaling there looks the
    same as upscaling here, at a sixth of the PNG encode time (~55 vs ~360 ms
    on the A65).
    """
    v = np.clip(heat, 0, 1).astype(np.float32)
    col = cv2.cvtColor(cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    a = (np.clip(v * 1.6, 0, 1) * 210).astype(np.uint8)
    return png_rgba(np.dstack([col, a]))


def mask_layer(mask: np.ndarray, hw: tuple[int, int], rgb: tuple[int, int, int], outline: bool = False) -> bytes:
    """A mask as a flat-colour RGBA layer; only alpha varies, built with OpenCV
    uint8 ops (numpy boolean indexing was ~150 ms per layer on the A65)."""
    h, w = hw
    m = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    if outline:
        edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        alpha = cv2.max(cv2.multiply(m, 70), cv2.multiply(edge, 255))
    else:
        alpha = cv2.multiply(m, 110)
    planes = [np.full((h, w), c, np.uint8) for c in (rgb[2], rgb[1], rgb[0])]   # BGR for imencode
    ok, buf = cv2.imencode(".png", cv2.merge(planes + [alpha]))
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


# --------------------------------------------------------------------------- model hub
class Hub:
    """Compiled models published on Hugging Face, and fetching one onto this board.

    The repository's `models.json` (tools/publish_hub.py) lists each model's MPK with its
    size and SHA-256. A download goes to `<models>/<file>.part` and is moved into place
    only once its size and sum are right, so the model list never sees half a file; a
    stopped download is kept and the next one continues from where it stopped.
    What comes from the network is checked: names and files against patterns, and
    every byte against the sum.
    """

    REFRESH = 300                                   # seconds a fetched list is good for
    NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,40}")
    FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,80}\.tar\.gz")
    SPARE = 1 << 30                                 # disk left free after a download

    def __init__(self, repo: str, models_dir: Path, loaded, catalog):
        self.repo, self.dir = repo, Path(models_dir)
        self._loaded, self._catalog = loaded, catalog
        self._lock = threading.Lock()
        self._models: dict | None = None
        self._fetched = 0.0
        self._fetching = False
        self._error: str | None = None
        self._job: dict | None = None
        self._failed: dict[str, str] = {}
        self.quiet = False                              # the --fetch command reports instead

    def url(self, path: str) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/main/{path}"

    def _refresh(self) -> None:
        try:
            req = urllib.request.Request(self.url("models.json"), headers={"User-Agent": "neat-change-studio"})
            with urllib.request.urlopen(req, timeout=15) as r:
                doc = json.loads(r.read(1 << 20))
            models = {}
            for name, e in doc["models"].items():
                files = e["files"]
                if not (self.NAME.fullmatch(name) and self.NAME.fullmatch(e["path"]) and len(files) == 1
                        and self.FILE.fullmatch(files[0]["name"]) and isinstance(files[0]["bytes"], int)
                        and re.fullmatch(r"[0-9a-f]{64}", files[0]["sha256"])):
                    raise ValueError(f"the entry for {name!r} is not one this app understands")
                models[name] = e
            self._models, self._error = models, None
        except (OSError, ValueError, KeyError, TypeError) as e:
            self._error = f"Could not read the model list from Hugging Face: {e}"
        finally:
            self._fetched, self._fetching = time.monotonic(), False

    def models(self, wait: bool = False) -> dict:
        """The list as last fetched; a fetch starts when there is none or it is old."""
        with self._lock:
            stale = time.monotonic() - self._fetched > (self.REFRESH if self._models is not None else 15)
            if stale and not self._fetching:
                self._fetching = True
                t = threading.Thread(target=self._refresh, name="hub-list", daemon=True)
                t.start()
                if wait:
                    t.join(20)
        return self._models or {}

    def _part(self, e: dict) -> Path:
        return self.dir / (e["files"][0]["name"] + ".part")

    def describe(self) -> dict:
        out, job = {}, self._job
        for name, e in self.models().items():
            f = e["files"][0]
            path = self.dir / f["name"]
            downloading = job is not None and job["name"] == name and job["thread"].is_alive()
            progress = None
            if downloading:
                secs = time.monotonic() - job["started"]
                rate = job["fetched"] / secs if secs > 1 else 0
                progress = {"fraction": round(job["done"] / f["bytes"], 4), "done_bytes": job["done"],
                            "total_bytes": f["bytes"], "bytes_per_second": int(rate),
                            "seconds_left": int((f["bytes"] - job["done"]) / rate) if rate else None}
            on_board = path.is_file() and path.stat().st_size == f["bytes"]
            part = self._part(e)
            out[name] = {
                "title": e.get("title", name), "about": e.get("about", ""), "precision": e.get("precision"),
                "latency_ms": e.get("latency_ms"), "accuracy": e.get("accuracy") or {},
                "file": f["name"], "bytes": f["bytes"], "url": self.url(f"{e['path']}/{f['name']}"),
                "partial_bytes": part.stat().st_size if part.is_file() and not downloading else 0,
                "state": "downloading" if downloading else "on_board" if on_board else "available",
                "loaded": name == self._loaded(), "progress": progress, "error": self._failed.get(name),
            }
        return {"repo": self.repo, "page": f"https://huggingface.co/{self.repo}",
                "collection": f"https://huggingface.co/collections/{HUB_COLLECTION}" if self.repo == HUB_REPO else None,
                "error": self._error, "listed": self._models is not None, "models": out,
                "directory": str(self.dir), "free_bytes": shutil.disk_usage(self.dir).free}

    def download(self, name: str) -> None:
        e = self.models().get(name)
        if e is None:
            raise CywsError(f"Hugging Face lists no model named {name!r}")
        f = e["files"][0]
        with self._lock:
            if self._job is not None and self._job["thread"].is_alive():
                raise CywsError(f"{self._job['name']} is downloading; one download at a time")
            path = self.dir / f["name"]
            if path.is_file() and path.stat().st_size == f["bytes"]:
                raise CywsError(f"{name} is already on this board")
            part = self._part(e)
            need = f["bytes"] - (part.stat().st_size if part.is_file() else 0)
            free = shutil.disk_usage(self.dir).free
            if free < need + self.SPARE:
                raise CywsError(f"Not enough disk for {name}: {need >> 20} MB to fetch, {free >> 20} MB free")
            self._failed.pop(name, None)
            job = {"name": name, "done": 0, "fetched": 0, "started": time.monotonic(), "cancel": threading.Event()}
            job["thread"] = threading.Thread(target=self._run, args=(job, e), name="hub-download", daemon=True)
            self._job = job
            job["thread"].start()

    def cancel(self, name: str) -> None:
        job = self._job
        if job is not None and job["name"] == name:
            job["cancel"].set()

    def discard(self, name: str) -> None:
        """Remove what a stopped download left behind."""
        e = self.models().get(name)
        if e is None:
            raise CywsError(f"Hugging Face lists no model named {name!r}")
        job = self._job
        if job is not None and job["name"] == name and job["thread"].is_alive():
            raise CywsError(f"{name} is downloading; cancel it first")
        self._part(e).unlink(missing_ok=True)
        self._failed.pop(name, None)

    def _run(self, job: dict, e: dict) -> None:
        name, f = job["name"], e["files"][0]
        target, part = self.dir / f["name"], self._part(e)
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            have = part.stat().st_size if part.is_file() else 0
            if have > f["bytes"]:
                part.unlink()
                have = 0
            digest = hashlib.sha256()
            if have:                                     # continue: hash what is already there
                with part.open("rb") as old:
                    while chunk := old.read(1 << 22):
                        digest.update(chunk)
            job["done"] = have
            headers = {"User-Agent": "neat-change-studio", **({"Range": f"bytes={have}-"} if have else {})}
            req = urllib.request.Request(self.url(f"{e['path']}/{f['name']}"), headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:             # the server sent everything: start over
                    have, digest, job["done"] = 0, hashlib.sha256(), 0
                with part.open("ab" if have else "wb") as out:
                    while chunk := r.read(1 << 20):
                        if job["cancel"].is_set():
                            raise InterruptedError()
                        out.write(chunk)
                        digest.update(chunk)
                        job["done"] += len(chunk)
                        job["fetched"] += len(chunk)
                    out.flush()
                    os.fsync(out.fileno())
            size = part.stat().st_size
            if size != f["bytes"] or digest.hexdigest() != f["sha256"]:
                part.unlink(missing_ok=True)
                raise ValueError(f"{f['name']} arrived damaged ({size} of {f['bytes']} bytes); try again")
            os.replace(part, target)
            if not self.quiet:
                log(f"downloaded {name} from {self.repo} ({f['bytes'] >> 20} MB)")
        except InterruptedError:
            self._failed[name] = "Download cancelled. What arrived is kept; Download continues from there."
        except (OSError, ValueError) as err:
            self._failed[name] = f"The download stopped: {err}"
            log(f"download of {name} failed: {err}")


# --------------------------------------------------------------------------- demo state
class Demo:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.models_dir = Path(args.models).resolve()
        self.lock = threading.RLock()          # one pipeline run / model switch at a time
        self.backend = None
        self.model_name: str | None = None
        self.state, self.error = "starting", ""
        # model switch progress, for the page's animation: stage is release | load | warm
        self.load_stage, self.load_target, self.load_t0 = "", "", 0.0
        self.last_load_s: dict[str, float] = {}
        self.base_cfg = self._base_config()
        self.results: "OrderedDict[str, dict]" = OrderedDict()
        self.decoded: "OrderedDict[tuple, object]" = OrderedDict()
        self.decoded_lock = threading.Lock()
        self.render_pool = ThreadPoolExecutor(max_workers=5, thread_name_prefix="cyws-render")
        self.rid = itertools.count(1)
        self.upload_dir = Path(args.uploads)
        shutil.rmtree(self.upload_dir, ignore_errors=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.uploads: "OrderedDict[str, object]" = OrderedDict()
        self.uid = itertools.count(1)
        self.pairs: dict[str, object] = {}
        self.pair_info: list[dict] = []
        self.thumbs: dict[tuple[str, str], bytes] = {}
        self.runs = 0
        self.mla_ms: list[float] = []
        repo = getattr(args, "hub_repo", "")
        self.hub = Hub(repo, self.models_dir, loaded=lambda: self.model_name, catalog=self._catalog) if repo else None
        self._load_dataset()

    def _base_config(self) -> AppConfig:
        """Default AppConfig, with the CLI's 'balanced' preset applied when the CLI defines presets."""
        cfg = AppConfig()
        try:
            import run_cyws_neat as cli
            presets = getattr(cli, "PRESETS", None)
            if isinstance(presets, dict) and "balanced" in presets:
                fields = {f.name for f in dataclasses.fields(AppConfig)}
                cfg = dataclasses.replace(cfg, **{k: v for k, v in presets["balanced"].items() if k in fields})
        except Exception as e:   # presets are a convenience; the dataclass defaults are fine
            log(f"note: no CLI presets ({e})")
        return cfg

    def presets(self) -> dict:
        try:
            import run_cyws_neat as cli
            p = getattr(cli, "PRESETS", None)
            if isinstance(p, dict):
                return {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()} for k, v in p.items()}
        except Exception:
            pass
        return {}

    # -- dataset ----------------------------------------------------------------
    def _load_dataset(self) -> None:
        root = Path(self.args.data)
        if not root.exists():
            log(f"dataset {root} not found; only uploads will work")
            return
        items, warnings = discover_pairs(root)
        for w in warnings:
            log(f"dataset: {w}")
        cache = Path(self.args.cache)
        cache.mkdir(parents=True, exist_ok=True)
        info_path = cache / "pairs.json"
        known = json.loads(info_path.read_text()) if info_path.exists() else {}
        for it in items:
            self.pairs[it.pair_id] = it
            row = known.get(it.pair_id)
            if row is None:
                row = {"id": it.pair_id, "has_gt": it.mask_path is not None, "gt_change": None}
                if it.mask_path is not None:
                    m = cv2.imread(str(it.mask_path), cv2.IMREAD_REDUCED_GRAYSCALE_4)
                    row["gt_change"] = bool(m is not None and (m > 127).mean() > 1e-4)
                known[it.pair_id] = row
            self.pair_info.append(row)
        info_path.write_text(json.dumps(known, indent=1))
        log(f"dataset: {len(self.pairs)} pairs from {root}")

    def thumb(self, pid: str, which: str) -> bytes:
        key = (pid, which)
        if key not in self.thumbs:
            path = Path(self.args.cache) / f"thumb_{pid}_{which}.jpg"
            if path.exists():
                self.thumbs[key] = path.read_bytes()
            else:
                it = self.pairs[pid] if pid in self.pairs else self.uploads[pid]
                src = it.after_path if which == "after" else it.before_path
                img = cv2.imread(str(src), cv2.IMREAD_REDUCED_COLOR_8)
                if img is None:
                    raise CywsError(f"cannot read {src}")
                h, w = img.shape[:2]
                s = 320 / max(h, w)
                img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                self.thumbs[key] = buf.tobytes()
                if pid in self.pairs:
                    path.write_bytes(self.thumbs[key])
        return self.thumbs[key]

    # -- model ------------------------------------------------------------------
    def _catalog(self) -> "OrderedDict[str, dict]":
        """Models this demo knows: the built-in two, plus any Hugging Face lists."""
        out = OrderedDict((n, dict(m)) for n, m in MODELS.items())
        for n, e in (self.hub.models() if self.hub else {}).items():
            label = MODELS[n]["label"] if n in MODELS else f"{n} · {e.get('title') or n}"
            out[n] = {"file": e["files"][0]["name"], "label": label, "bytes": e["files"][0]["bytes"]}
        return out

    def _on_board(self, m: dict) -> bool:
        path = self.models_dir / m["file"]
        return path.is_file() and ("bytes" not in m or path.stat().st_size == m["bytes"])

    def available_models(self) -> list[dict]:
        return [{"name": n, "label": m["label"], "available": self._on_board(m),
                 "loaded": n == self.model_name} for n, m in self._catalog().items()]

    def load_model(self, name: str) -> None:
        catalog = self._catalog()
        if name not in catalog:
            raise CywsError(f"unknown model {name!r}")
        path = self.models_dir / catalog[name]["file"]
        if not path.exists():
            err = CywsError(f"{name} is not on this board yet: download it in Settings (Model)")
            if self.backend is None:            # nothing loaded: say so instead of "starting" forever
                self.state, self.error = "error", str(err)
            raise err
        with self.lock:
            self.state, self.error = "loading", ""
            t0 = time.perf_counter()
            self.load_target, self.load_t0 = name, t0
            try:
                if self.backend is not None:          # close before building: one runner at a time
                    self.load_stage = "release"
                    self.backend.close()
                    self.backend, self.model_name = None, None
                self.load_stage = "load"
                if self.args.backend == "neat":
                    self.backend = backends.NeatBackend(str(path), max_in_flight=2)
                else:
                    self.backend = backends.make_backend(self.args.backend, self.args.onnx)
                # warm-up: first inference pays one-time costs
                self.load_stage = "warm"
                z = np.zeros((256, 256, 3), np.float32)
                self.backend.infer(z, z)
                self.model_name, self.state = name, "ready"
                self.mla_ms.clear()
                self.last_load_s[name] = time.perf_counter() - t0
                log(f"model {name} ready ({self.last_load_s[name]:.1f} s)")
            except Exception as e:
                self.state, self.error = "error", f"{type(e).__name__}: {e}"
                log(f"model load failed: {self.error}")
                raise
            finally:
                self.load_stage = ""

    def close(self) -> None:
        with self.lock:
            if self.backend is not None:
                try:
                    self.backend.close()
                finally:
                    self.backend, self.model_name = None, None

    # -- uploads ----------------------------------------------------------------
    def add_upload(self, before_fs, after_fs, before_from: str | None = None) -> dict:
        """Store an uploaded pair. With before_from (an earlier upload's id), the before
        photo is copied from that upload instead, so a live webcam only sends new frames."""
        src = None
        if before_fs is None:
            src = self.uploads.get(before_from or "")
            if src is None:
                raise CywsError("the before photo has expired; capture it again")
        uid = f"upload{next(self.uid)}"
        d = self.upload_dir / uid
        d.mkdir(parents=True)
        paths = {}
        if src is not None:
            paths["before"] = d / f"before{src.before_path.suffix}"
            shutil.copyfile(src.before_path, paths["before"])
        for role, fs in (("before", before_fs), ("after", after_fs)):
            if fs is None:
                continue
            ext = Path(fs.filename or "").suffix.lower() or ".jpg"
            if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"):
                raise CywsError(f"{role}: unsupported file type {ext!r}")
            paths[role] = d / f"{role}{ext}"
            fs.save(paths[role])
        item = static_pair(paths["before"], paths["after"], pair_id=uid)
        # validate now, so the page gets a clear message instead of a failed run later
        # (and no server paths in it)
        for role, path in (("before", item.before_path), ("after", item.after_path)):
            if role == "before" and src is not None:
                continue                                   # already checked when it was uploaded
            try:
                ingestion.read_image_rgb(path)
            except Exception as e:
                log(f"upload {uid}: {role} unreadable: {e}")
                shutil.rmtree(d, ignore_errors=True)
                raise CywsError(f"the {role} photo isn't a readable image") from e
        self.uploads[uid] = item
        while len(self.uploads) > MAX_UPLOADS:
            old, _ = self.uploads.popitem(last=False)
            shutil.rmtree(self.upload_dir / old, ignore_errors=True)
            for k in [k for k in self.thumbs if k[0] == old]:
                del self.thumbs[k]
        return {"id": uid, "before": fs_name(before_fs) if before_fs else before_from, "after": fs_name(after_fs)}

    # -- run --------------------------------------------------------------------
    def config_from(self, opts: dict) -> AppConfig:
        cfg = self.base_cfg
        fields = {f.name for f in dataclasses.fields(AppConfig)}
        preset = opts.get("preset")
        if preset:
            p = self.presets().get(preset)
            if p is None:
                raise CywsError(f"unknown preset {preset!r}")
            cfg = dataclasses.replace(cfg, **{k: (tuple(v) if isinstance(v, list) else v)
                                              for k, v in p.items() if k in fields})
        upd = {}
        for k, spec in OPTION_SPEC.items():
            if k not in opts or opts[k] is None:
                continue
            v = opts[k]
            if spec[0] is bool:
                upd[k] = bool(v)
            elif spec[0] is float:
                v = float(v)
                if not spec[1] <= v <= spec[2]:
                    raise CywsError(f"{k} must be in [{spec[1]}, {spec[2]}]")
                upd[k] = v
            else:
                if str(v) not in spec[1]:
                    raise CywsError(f"{k} must be one of {spec[1]}")
                upd[k] = tuple(int(x) for x in str(v).split(",")) if k == "levels" else str(v)
        if upd.get("mask_mode") == "diff" and not upd.get("register", cfg.register):
            upd["mask_mode"] = "box"          # diff masks need the aligned before image
        return dataclasses.replace(cfg, **{k: v for k, v in upd.items() if k in fields})

    def run(self, pid: str, opts: dict) -> dict:
        item = self.pairs.get(pid) or self.uploads.get(pid)
        if item is None:
            raise CywsError(f"unknown pair {pid!r}")
        cfg = self.config_from(opts)
        pair, views = self._load(pid, item, cfg)  # CPU only: decodes outside the MLA lock
        with self.lock:
            if self.state != "ready" or self.backend is None:
                raise CywsError(f"model is not ready ({self.state}{': ' + self.error if self.error else ''})")
            app = CywsApp(self.backend, cfg)
            res = app.process(pair)
            model = self.model_name
            self.runs += 1
        t0 = time.perf_counter()
        H, W = pair.after.shape[:2]
        vh, vw = _fit_hw(H, W)
        # Each display image is encoded on its own worker (cv2 releases the GIL while
        # encoding). The photo and GT layers only depend on the decoded pair, so
        # they are encoded once per pair and reused on re-runs with new settings.
        jobs = {
            "heat": lambda: heat_layer(res.heat),
            # half the display size: PNG encoding is ~4x cheaper (~30 vs ~115 ms on the A65),
            # and this translucent tint sits under crisp SVG box outlines anyway
            "mask": lambda: mask_layer(res.mask, ((vh + 1) // 2, (vw + 1) // 2), (255, 64, 160)),
        }
        if "after" not in views:
            jobs["after"] = lambda: jpeg(_fit(pair.after))
        if pair.gt_mask is not None and "gt" not in views:
            jobs["gt"] = lambda: mask_layer(pair.gt_mask, (vh, vw), (40, 220, 255), outline=True)
        if res.before_aligned is not None:  # registration output: depends on this run's settings
            jobs["before_run"] = lambda: jpeg(_fit(res.before_aligned))
        elif "before" not in views:
            jobs["before"] = lambda: jpeg(_fit(ingestion.resize_to(pair.before, W, H)))
        futures = {k: self.render_pool.submit(f) for k, f in jobs.items()}
        metrics = None
        if pair.gt_mask is not None:
            metrics = post_process.metrics(res.mask, pair.gt_mask, res.detections)
        done = {k: f.result() for k, f in futures.items()}
        for k in ("after", "gt", "before"):
            if k in done:
                views[k] = done[k]
        images = {"after": views["after"], "before": done.get("before_run", views.get("before")),
                  "heat": done["heat"], "mask": done["mask"]}
        if pair.gt_mask is not None:
            images["gt"] = views["gt"]
        rid = f"r{next(self.rid)}"
        self.results[rid] = images
        while len(self.results) > MAX_RESULTS:
            self.results.popitem(last=False)
        windows = [{"id": w.id, "level": w.level, "x0": w.x0, "y0": w.y0, "w": w.w, "h": w.h} for w in res.windows]
        scale = (pair.meta.get("scale") or [1.0, 1.0])
        out = {
            "rid": rid, "pair": pid, "model": model,
            "size": {"w": W, "h": H}, "view": {"w": vw, "h": vh},
            "orig_size": pair.meta.get("orig_size"),
            "scale": scale,
            "detections": [d.to_dict() for d in res.detections],
            "counts": res.meta.get("counts", {}),
            "windows": windows,
            "registration": None if res.registration is None else {
                "method": res.registration.method, "ok": res.registration.ok,
                "inliers": res.registration.inliers, "reason": res.registration.reason},
            "timings_ms": res.timings_ms,
            "render_ms": round(1000 * (time.perf_counter() - t0), 1),
            "metrics": None if metrics is None else post_process_scalars(metrics),
            "images": {k: f"/api/result/{rid}/{k}" for k in images},
            "config": {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)},
        }
        stats = self.backend.stats() if self.backend is not None else {}
        lat = (stats.get("latency") or {}).get("mean_ms")
        if lat:
            self.mla_ms.append(float(lat))
            self.mla_ms[:] = self.mla_ms[-50:]
        return json.loads(json.dumps(out, default=_json_default))

    def _load(self, pid: str, item, cfg: AppConfig):
        """Decoded pair for (pid, decode settings), from a small LRU when possible.

        Only work_long, register and allow_squash change what load_pair returns;
        a cached pair reports a load time of 0.
        """
        key = (pid, str(item.after_path), str(item.before_path), cfg.work_long, cfg.register, cfg.allow_squash)
        with self.decoded_lock:
            pair = self.decoded.get(key)
            if pair is not None:
                self.decoded.move_to_end(key)
        if pair is not None:
            pair, views = pair
            return dataclasses.replace(pair, meta={**pair.meta, "load_ms": 0.0, "load_cached": True}), views
        pair = CywsApp(None, cfg).load(item)
        views: dict = {}  # encoded display images that depend only on the decoded pair
        with self.decoded_lock:
            self.decoded[key] = (pair, views)
            while len(self.decoded) > MAX_DECODED:
                self.decoded.popitem(last=False)
        return pair, views

    def status(self) -> dict:
        stats = {}
        try:
            stats = self.backend.stats() if self.backend is not None else {}
        except Exception:
            pass
        return {"app": "cyws-demo", "state": self.state, "error": self.error, "model": self.model_name,
                "models": self.available_models(), "backend": self.args.backend, "runs": self.runs,
                "pairs": len(self.pairs), "uploads": list(self.uploads), "presets": self.presets(),
                "defaults": {k: _json_default(getattr(self.base_cfg, k)) if isinstance(getattr(self.base_cfg, k), tuple)
                             else getattr(self.base_cfg, k) for k in OPTION_SPEC if hasattr(self.base_cfg, k)},
                "latency": stats.get("latency"), "https_port": self.args.port if self.args.tls else None,
                "load": {"stage": self.load_stage, "target": self.load_target,
                         "elapsed_s": round(time.perf_counter() - self.load_t0, 2) if self.load_stage else 0.0,
                         "expected_s": self.last_load_s.get(self.load_target)}}


def post_process_scalars(m: dict) -> dict:
    keys = ("iou", "precision", "recall", "f1", "gt_components", "gt_components_hit", "component_recall",
            "num_boxes", "box_precision")
    return {k: m.get(k) for k in keys if k in m}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (tuple, set)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def fs_name(fs) -> str:
    return Path(fs.filename or "").name


# --------------------------------------------------------------------------- web
def create_app(demo: Demo) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

    def fail(e: Exception, code: int = 400):
        msg = str(e.args[0]) if isinstance(e, KeyError) and e.args else f"{e}"
        return jsonify({"error": msg[:1].upper() + msg[1:]}), code   # shown as is on the page

    @app.get("/")
    def index():
        resp = send_from_directory(WEB, "index.html")
        resp.headers["Cache-Control"] = "no-cache"          # always revalidate the page shell
        return resp

    @app.get("/static/<path:name>")
    def static_file(name):
        resp = send_from_directory(WEB, name)
        # fonts and logos never change; the page's own css/js revalidate (ETag) so updates show at once
        resp.headers["Cache-Control"] = "public, max-age=604800, immutable" if name.startswith("assets/") else "no-cache"
        return resp

    @app.get("/api/status")
    def status():
        return jsonify(demo.status())

    @app.get("/api/pairs")
    def pairs():
        return jsonify({"pairs": demo.pair_info, "uploads": list(demo.uploads)})

    @app.get("/api/thumb/<pid>/<which>")
    def thumb(pid, which):
        if which not in ("before", "after"):
            return fail(ValueError("which must be before|after"))
        try:
            return Response(demo.thumb(pid, which), mimetype="image/jpeg",
                            headers={"Cache-Control": "max-age=86400"})
        except KeyError:
            return fail(KeyError(f"unknown pair {pid}"), 404)
        except Exception as e:
            return fail(e, 500)

    @app.post("/api/run")
    def run():
        body = request.get_json(silent=True) or {}
        try:
            return jsonify(demo.run(str(body.get("id", "")), body.get("options") or {}))
        except CywsError as e:
            return fail(e)
        except Exception as e:
            log(traceback.format_exc())
            return fail(e, 500)

    @app.get("/api/result/<rid>/<name>")
    def result_image(rid, name):
        imgs = demo.results.get(rid)
        if imgs is None or name not in imgs:
            return fail(KeyError("result expired; run again"), 404)
        mt = "image/jpeg" if name in ("after", "before") else "image/png"
        return Response(imgs[name], mimetype=mt, headers={"Cache-Control": "max-age=3600"})

    @app.post("/api/upload")
    def upload():
        b, a = request.files.get("before"), request.files.get("after")
        before_from = request.form.get("before_from")
        if not a or not (b or before_from):
            return fail(ValueError("send both 'before' and 'after' files"))
        try:
            return jsonify(demo.add_upload(b, a, before_from=None if b else before_from))
        except Exception as e:
            return fail(e)

    @app.get("/api/hub")
    def hub_list():
        if demo.hub is None:
            return jsonify({"repo": None, "listed": True, "models": {}, "error": None})
        demo.hub.models(wait=demo.hub._models is None)        # the first visit waits for the list
        return jsonify(demo.hub.describe())

    @app.post("/api/hub/<action>")
    def hub_action(action):
        name = str((request.get_json(silent=True) or {}).get("name", ""))
        if demo.hub is None:
            return fail(CywsError("this demo has no model hub (started with --hub-repo '')"))
        fn = {"download": demo.hub.download, "cancel": demo.hub.cancel, "discard": demo.hub.discard}.get(action)
        if fn is None:
            return fail(KeyError(f"unknown action {action}"), 404)
        try:
            fn(name)
            return jsonify(demo.hub.describe())
        except Exception as e:
            return fail(e)

    @app.post("/api/model")
    def model():
        name = (request.get_json(silent=True) or {}).get("name", "")
        try:
            demo.load_model(name)
            return jsonify(demo.status())
        except Exception as e:
            return fail(e)

    return app


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8098)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--model", default="bf16", choices=list(MODELS))
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--data", default=os.environ.get("CYWS_DATA") or str(ROOT / "data/testCYWS"))
    ap.add_argument("--cache", default=str(ROOT / "webapp/cache"))
    ap.add_argument("--uploads", default=str(ROOT / "webapp/uploads"))
    ap.add_argument("--backend", default="neat", choices=["neat", "onnx", "fake"],
                    help="neat on the board; onnx/fake for trying the page on a PC")
    ap.add_argument("--onnx", default=str(ROOT / "build/onnx/cyws_fused.onnx"))
    ap.add_argument("--hub-repo", default=HUB_REPO,
                    help="Hugging Face repository whose models.json lists downloadable models ('' for none)")
    ap.add_argument("--list-hub", action="store_true", help="list the models on Hugging Face and exit")
    ap.add_argument("--fetch", nargs="+", metavar="NAME",
                    help="download these models from Hugging Face into --models and exit ('all' for every one)")
    ap.add_argument("--no-https", dest="https", action="store_false",
                    help="serve plain HTTP only (by default the port also speaks HTTPS when a certificate "
                         "exists; browsers only allow the webcam on HTTPS pages)")
    ap.add_argument("--cert", default=str(ROOT / "webapp/certs/cert.pem"))
    ap.add_argument("--key", default=str(ROOT / "webapp/certs/key.pem"))
    args = ap.parse_args()
    if args.list_hub or args.fetch:
        return hub_command(args)
    faulthandler.register(signal.SIGUSR1)

    demo = Demo(args)
    threading.Thread(target=lambda: _safe_load(demo, args.model), name="model-load", daemon=True).start()
    flask_app = create_app(demo)
    server = make_server(args.host, args.port, flask_app, threaded=True)
    args.tls = False
    if args.https:
        if Path(args.cert).exists() and Path(args.key).exists():
            serve_both_protocols(server, args.cert, args.key)
            args.tls = True
        else:
            log(f"no certificate at {args.cert}; HTTPS (and so the webcam) is off")
    stop = threading.Event()

    def shutdown(signum, _frame):
        log(f"signal {signum}: shutting down")
        stop.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, shutdown)
    ip = socket.gethostbyname(socket.gethostname()) if args.host == "0.0.0.0" else args.host
    log(f"CYWS demo at http://{ip}:{args.port}/  (backend {args.backend})")
    if args.tls:
        log(f"same port over HTTPS, for the webcam: https://{ip}:{args.port}/  (self-signed certificate)")
    try:
        server.serve_forever()
    finally:
        faulthandler.dump_traceback_later(20, exit=True)   # never hang on shutdown
        demo.close()
        log("closed")
    return 0


def serve_both_protocols(server, cert: str, key: str) -> None:
    """Let one port speak both HTTP and HTTPS.

    Each connection is sniffed in its own handler thread: a TLS handshake starts with
    byte 0x16, which no HTTP request line does, so those connections are wrapped in TLS
    and the rest are served as plain HTTP.
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    plain_finish = server.finish_request

    def finish_request(request, client_address):
        try:
            request.settimeout(60)                     # browsers open idle spare connections
            first = request.recv(1, socket.MSG_PEEK)
        except OSError:
            return
        if first != b"\x16":
            request.settimeout(None)
            plain_finish(request, client_address)
            return
        try:
            tls = ctx.wrap_socket(request, server_side=True)
        except (ssl.SSLError, OSError):                # e.g. the browser rejected the self-signed cert
            return
        tls.settimeout(None)
        try:
            plain_finish(tls, client_address)
        except ssl.SSLError:
            pass
        finally:
            try:
                tls.close()
            except OSError:
                pass

    server.finish_request = finish_request


def hub_command(args) -> int:
    """`--list-hub` and `--fetch`: the model hub from a terminal. Nothing is served and
    the MLA is not touched, so it works while the demo runs."""
    if not args.hub_repo:
        print("no --hub-repo: there is nothing to list or fetch", file=sys.stderr)
        return 1
    models_dir = Path(args.models).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    hub = Hub(args.hub_repo, models_dir, loaded=lambda: None, catalog=lambda: {})
    hub.quiet = True
    hub._refresh()
    listed = hub.describe()
    if listed["error"]:
        print(listed["error"], file=sys.stderr)
        return 1
    mb = lambda n: f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"
    if args.list_hub:
        print(f"models in {listed['page']} (into {models_dir}):")
        for name, m in listed["models"].items():
            print(f"  {name:8s} {m['title']:14s} {mb(m['bytes']):>8s}  {m['state'].replace('_', ' ')}")
        return 0
    names = list(listed["models"]) if args.fetch == ["all"] else args.fetch
    failed = 0
    for name in names:
        m = listed["models"].get(name)
        if m is None:
            print(f"  {name}: not in {args.hub_repo} (there: {', '.join(listed['models'])})", file=sys.stderr)
            failed += 1
            continue
        if m["state"] == "on_board":
            print(f"  {name}: already here ({models_dir / m['file']})")
            continue
        try:
            hub.download(name)
        except CywsError as e:
            print(f"  {name}: {e}", file=sys.stderr)
            failed += 1
            continue
        job, shown = hub._job, -1
        try:
            while job["thread"].is_alive():
                pr = hub.describe()["models"][name]["progress"]
                if pr and sys.stdout.isatty():
                    print(f"\r\033[K  {name}: {mb(pr['done_bytes'])} of {mb(pr['total_bytes'])} · {pr['fraction'] * 100:.0f}%"
                          f" · {pr['bytes_per_second'] / 1e6:.1f} MB/s", end="", flush=True)
                elif pr and int(pr["fraction"] * 10) > shown:
                    shown = int(pr["fraction"] * 10)
                    print(f"  {name}: {pr['fraction'] * 100:.0f}% of {mb(pr['total_bytes'])}", flush=True)
                job["thread"].join(0.5)
        except KeyboardInterrupt:
            hub.cancel(name)
            job["thread"].join()
            print(f"\n  {name}: cancelled; what arrived is kept and the next --fetch continues it")
            return 130
        if sys.stdout.isatty():
            print("\r\033[K", end="")
        if hub.describe()["models"][name]["state"] == "on_board":
            print(f"  {name}: downloaded and checked ({models_dir / m['file']})")
        else:
            print(f"  {name}: {hub._failed.get(name, 'the download stopped')}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


def _safe_load(demo: Demo, name: str) -> None:
    try:
        demo.load_model(name)
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())
