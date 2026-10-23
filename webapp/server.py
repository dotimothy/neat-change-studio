#!/usr/bin/env python3
"""Neat Change Studio: change-detection web demo on the SiMa Modalix MLA.

Pick a before/after pair from the test set (or upload your own two photos); the
loaded model runs on the MLA through pyneat via the app in ../app: CYWS (trained,
per 256x256 window pair) or DINOv2 (zero-shot, per 448x448 image; app/families.py
says how each family loads and runs). The page shows the detected changes on the "after" image together
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
import backends  # noqa: E402,F401 - families.py builds backends through it (tests patch server.backends)
import families  # noqa: E402
import ingestion  # noqa: E402
import post_process  # noqa: E402
from app import CywsApp  # noqa: E402,F401 - kept for callers that import it from here
from common import AppConfig, CywsError  # noqa: E402
from dataset_loader import discover_pairs, static_pair  # noqa: E402

mimetypes.add_type("font/woff2", ".woff2")      # not in every platform's mime table
mimetypes.add_type("image/svg+xml", ".svg")
mimetypes.add_type("application/manifest+json", ".webmanifest")

VIEW_LONG = 1600          # long side of the images sent to the browser
MAX_RESULTS = 24          # rendered results kept in memory
MAX_DECODED = 6           # decoded pairs kept, so re-runs with new settings skip the JPEG decode
MAX_UPLOADS = 20
MODELS = OrderedDict([
    ("bf16", {"file": "cyws_bf16_mpk.tar.gz", "label": "bf16 · Most Faithful", "family": "cyws", "title": "CYWS bf16"}),
    ("w8a16", {"file": "cyws_w8a16_mpk.tar.gz", "label": "w8a16 · Faster", "family": "cyws", "title": "CYWS w8a16"}),
    ("dinov2-448", {"file": "dinov2_vits14_reg_448_bf16_mpk.tar.gz", "label": "DINOv2 448 · Zero-Shot",
                    "family": "dinov2", "title": "DINOv2 448"}),
])


def models_storage(models_dir: Path) -> dict:
    """What the models take on disk: finished MPKs and stopped downloads."""
    files = sorted(Path(models_dir).glob("*_mpk.tar.gz")) if Path(models_dir).is_dir() else []
    parts = sorted(Path(models_dir).glob("*_mpk.tar.gz.part")) if Path(models_dir).is_dir() else []
    return {"used_bytes": sum(p.stat().st_size for p in files + parts), "model_count": len(files),
            "partial_count": len(parts)}


def _free_bytes(path: Path) -> int:
    """Free space where `path` is, or would be once created."""
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free


def default_models_dir() -> Path:
    """Where the compiled models live, shared by every install and kept across updates (like Neat
    GenAI Studio's /media/nvme/llima/models): CYWS_MODELS_DIR, else what setup.sh --models-dir
    remembered in .models-dir, else /media/nvme/models/cyws on a DevKit, else models/ here."""
    env = os.environ.get("CYWS_MODELS_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    remembered = ROOT / ".models-dir"
    if remembered.is_file() and remembered.read_text().strip():
        return Path(remembered.read_text().strip()).expanduser()
    if Path("/media/nvme").is_dir():
        return Path("/media/nvme/models/cyws")
    return ROOT / "models"


# compiled models on Hugging Face (tools/publish_hub.py), one repository per family
HUB_REPOS = ["dotimothy/sima-cyws", "dotimothy/sima-dinov2"]
HUB_REPO = HUB_REPOS[0]
HUB_COLLECTION = "dotimothy/simaai-change-detection-models-6ada4e701e447e9db528596b"

# CYWS run settings the page may change (each family has its own: families.py)
OPTION_SPEC = families.CYWS_OPTION_SPEC


def parse_repos(value) -> list[str]:
    """--hub-repo: a comma-separated list (or a list); '' means none."""
    items = value.split(",") if isinstance(value, str) else list(value or [])
    out = []
    for r in (str(x).strip() for x in items):
        if r and r not in out:
            out.append(r)
    return out


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

    Each repository's `models.json` (tools/publish_hub.py) lists models with their MPK,
    size and SHA-256, and the family each belongs to (an entry's own `family`, else the
    document's, else cyws). Several repositories are read in order; the first one wins
    a name clash. A download goes to `<models>/<file>.part` and is moved into place only
    once its size and sum are right, so the model list never sees half a file; a stopped
    download is kept and the next one continues from where it stopped.
    What comes from the network is checked: names and files against patterns, and every
    byte against the sum. An entry this app cannot use is skipped (listed in `skipped`),
    and the rest of the list still works.
    """

    REFRESH = 300                                   # seconds a fetched list is good for
    NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,40}")
    FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,80}\.tar\.gz")
    SPARE = 1 << 30                                 # disk left free after a download
    FAMILIES = ("cyws", "dinov2")

    def __init__(self, repo, models_dir: Path, loaded, catalog):
        self.repos = parse_repos(repo)
        self.repo = self.repos[0] if self.repos else ""
        self.dir = Path(models_dir)
        self._loaded, self._catalog = loaded, catalog
        self._lock = threading.Lock()
        self._models: dict | None = None
        self._by_repo: dict[str, dict] = {}             # repo -> its valid entries, as last fetched
        self._repo_error: dict[str, str | None] = {}
        self._skipped: list[dict] = []
        self._fetched = 0.0
        self._fetching = False
        self._error: str | None = None
        self._job: dict | None = None
        self._failed: dict[str, str] = {}
        self._verified: dict[str, tuple] = {}           # name -> (size, mtime_ns, sum matched)
        self.quiet = False                              # the --fetch command reports instead

    def url(self, path: str, repo: str | None = None) -> str:
        return f"https://huggingface.co/{repo or self.repo}/resolve/main/{path}"

    @classmethod
    def check_entry(cls, name, e, doc_family) -> tuple[dict | None, str]:
        """(normalised entry, '') or (None, why it is skipped)."""
        try:
            files = e["files"]
            if not (isinstance(name, str) and cls.NAME.fullmatch(name) and isinstance(e.get("path"), str)
                    and cls.NAME.fullmatch(e["path"]) and isinstance(files, list) and len(files) == 1
                    and isinstance(files[0].get("name"), str) and cls.FILE.fullmatch(files[0]["name"])
                    and isinstance(files[0].get("bytes"), int) and not isinstance(files[0]["bytes"], bool)
                    and isinstance(files[0].get("sha256"), str)
                    and re.fullmatch(r"[0-9a-f]{64}", files[0]["sha256"])):
                return None, "name, path or file is not one this app understands"
        except (KeyError, TypeError, AttributeError):
            return None, "name, path or file is not one this app understands"
        family = e.get("family") or doc_family or "cyws"
        if family not in cls.FAMILIES:
            return None, f"family {family!r} is not one this app runs"
        inp = e.get("input")
        if inp is not None:
            if not isinstance(inp, dict):
                return None, "input is not an object"
            size, images = inp.get("size"), inp.get("images")
            if size is not None and (not isinstance(size, int) or isinstance(size, bool)
                                     or (family == "dinov2" and (size < 14 or size % 14))
                                     or (family == "cyws" and size != 256)):
                return None, f"input size {size!r} does not fit a {family} model"
            if images is not None and images != (1 if family == "dinov2" else 2):
                return None, f"a {family} model does not take {images!r} images"
        card = e.get("card")
        if card is not None and not isinstance(card, dict):
            return None, "card is not an object"
        out = dict(e)
        out["family"] = family
        if card is not None:
            out["card"] = {str(k)[:60]: str(v)[:400] for k, v in card.items() if isinstance(v, (str, int, float))}
        return out, ""

    def _fetch(self, repo: str) -> dict:
        req = urllib.request.Request(self.url("models.json", repo), headers={"User-Agent": "neat-change-studio"})
        with urllib.request.urlopen(req, timeout=15) as r:
            doc = json.loads(r.read(1 << 20))
        if not isinstance(doc, dict) or not isinstance(doc.get("models"), dict):
            raise ValueError("models.json has no model list")
        if doc.get("version", 1) not in (1, 2):
            raise ValueError(f"models.json version {doc.get('version')!r} is newer than this app")
        return doc

    def _refresh(self) -> None:
        try:
            skipped: list[dict] = []
            for repo in self.repos:
                try:
                    doc = self._fetch(repo)
                except (OSError, ValueError, KeyError, TypeError) as e:
                    self._repo_error[repo] = f"Could not read the model list from Hugging Face ({repo}): {e}"
                    continue
                models = {}
                for name, e in doc["models"].items():
                    entry, why = self.check_entry(name, e, doc.get("family"))
                    if entry is None:
                        skipped.append({"repo": repo, "name": str(name), "reason": why})
                        continue
                    entry["repo"] = repo
                    models[name] = entry
                self._by_repo[repo], self._repo_error[repo] = models, None
            merged: dict = {}
            for repo in self.repos:                      # first repository wins a name clash
                for name, e in self._by_repo.get(repo, {}).items():
                    if name in merged:
                        skipped.append({"repo": repo, "name": name,
                                        "reason": f"{merged[name]['repo']} already lists a model named {name}"})
                        continue
                    merged[name] = e
            self._skipped = skipped
            if self._by_repo:
                self._models = merged
            self._error = next((self._repo_error[r] for r in self.repos if self._repo_error.get(r)), None)
        finally:
            self._fetched, self._fetching = time.monotonic(), False

    def _all_listed(self) -> bool:
        return all(r in self._by_repo for r in self.repos)

    def _all_answered(self) -> bool:
        """Every repository has answered, with its list or with an error."""
        return all(r in self._by_repo or self._repo_error.get(r) for r in self.repos)

    def models(self, wait: bool = False) -> dict:
        """The list as last fetched; a fetch starts when there is none or it is old."""
        with self._lock:
            # A repository that answered with an error (say, not published yet) waits as long
            # as a listed one; only a board with no list at all asks again soon.
            ttl = self.REFRESH if self._models is not None and self._all_answered() else 15
            stale = time.monotonic() - self._fetched > ttl
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
            st = path.stat() if path.is_file() else None
            on_board = st is not None and st.st_size == f["bytes"]
            mark = self._verified.get(name)
            verified = mark[2] if st is not None and mark and mark[:2] == (st.st_size, st.st_mtime_ns) else None
            part = self._part(e)
            if downloading:
                state = "downloading"
            elif on_board and verified is not False:
                state = "on_board"
            elif st is not None:
                state = "damaged"                   # there, but not the published file: re-download it
            else:
                state = "available"
            repo = e.get("repo") or self.repo
            out[name] = {
                "title": e.get("title", name), "about": e.get("about", ""), "precision": e.get("precision"),
                "latency_ms": e.get("latency_ms"), "accuracy": e.get("accuracy") or {},
                "family": e.get("family", "cyws"), "repo": repo, "card": e.get("card") or {},
                "input": e.get("input"), "facts": e.get("facts"),
                "file": f["name"], "bytes": f["bytes"], "url": self.url(f"{e['path']}/{f['name']}", repo),
                "partial_bytes": part.stat().st_size if part.is_file() and not downloading else 0,
                "state": state, "on_disk_bytes": st.st_size if st is not None else 0, "verified": verified,
                "replacing": downloading and st is not None,
                "loaded": name == self._loaded(), "progress": progress, "error": self._failed.get(name),
            }
        repos = [{"repo": r, "page": f"https://huggingface.co/{r}", "listed": r in self._by_repo,
                  "error": self._repo_error.get(r)} for r in self.repos]
        return {"repo": self.repo, "page": f"https://huggingface.co/{self.repo}",
                "collection": (f"https://huggingface.co/collections/{HUB_COLLECTION}"
                               if any(r in HUB_REPOS for r in self.repos) else None),
                "error": self._error, "listed": self._all_listed(), "models": out,
                "repos": repos, "skipped": list(self._skipped),
                "directory": str(self.dir), "free_bytes": _free_bytes(self.dir), **models_storage(self.dir)}

    def download(self, name: str, replace: bool = False) -> None:
        """Fetch a model in the background. With `replace`, also when it is already here
        (Re-download): the new copy goes to the .part file and replaces the old one only
        once its sum is right, so a failed re-download leaves the old file as it was."""
        e = self.models().get(name)
        if e is None:
            raise CywsError(f"Hugging Face lists no model named {name!r}")
        f = e["files"][0]
        with self._lock:
            if self._job is not None and self._job["thread"].is_alive():
                raise CywsError(f"{self._job['name']} is downloading; one download at a time")
            path = self.dir / f["name"]
            if not replace and path.is_file() and path.stat().st_size == f["bytes"]:
                raise CywsError(f"{name} is already on this board")
            part = self._part(e)
            need = f["bytes"] - (part.stat().st_size if part.is_file() else 0)
            free = _free_bytes(self.dir)
            if free < need + self.SPARE:
                raise CywsError(f"Not enough disk for {name}: {need >> 20} MB to fetch, {free >> 20} MB free")
            self._failed.pop(name, None)
            job = {"name": name, "done": 0, "fetched": 0, "started": time.monotonic(), "cancel": threading.Event()}
            job["thread"] = threading.Thread(target=self._run, args=(job, e), name="hub-download", daemon=True)
            self._job = job
            job["thread"].start()

    def redownload(self, name: str) -> None:
        self.download(name, replace=True)

    def verify(self, name: str) -> bool:
        """Check the file on the board against the published SHA-256 (about a second on the A65)."""
        e = self.models().get(name)
        if e is None:
            raise CywsError(f"Hugging Face lists no model named {name!r}")
        f = e["files"][0]
        path = self.dir / f["name"]
        if not path.is_file():
            raise CywsError(f"{name} is not on this board")
        st = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as file:
            while chunk := file.read(1 << 22):
                digest.update(chunk)
        ok = st.st_size == f["bytes"] and digest.hexdigest() == f["sha256"]
        self._verified[name] = (st.st_size, st.st_mtime_ns, ok)
        if ok:
            self._failed.pop(name, None)
        else:
            self._failed[name] = (f"{f['name']} on the board does not match the published file "
                                  f"({st.st_size} of {f['bytes']} bytes, or a different sum). Re-download it.")
        return ok

    def forget(self, name: str) -> None:
        """A model was deleted: drop what was known about its file."""
        self._verified.pop(name, None)
        self._failed.pop(name, None)

    def busy_with(self, name: str) -> bool:
        job = self._job
        return job is not None and job["name"] == name and job["thread"].is_alive()

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
            req = urllib.request.Request(self.url(f"{e['path']}/{f['name']}", e.get("repo")), headers=headers)
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
            st = target.stat()
            self._verified[name] = (st.st_size, st.st_mtime_ns, True)
            if not self.quiet:
                log(f"downloaded {name} from {e.get('repo') or self.repo} ({f['bytes'] >> 20} MB)")
        except InterruptedError:
            self._failed[name] = "Download cancelled. What arrived is kept; Download continues from there."
        except (OSError, ValueError) as err:
            self._failed[name] = f"The download stopped: {err}"
            log(f"download of {name} failed: {err}")


# --------------------------------------------------------------------------- demo state
class Demo:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.models_dir = Path(args.models).expanduser().resolve()
        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            log(f"cannot create the models directory {self.models_dir}: {e}")
        self.lock = threading.RLock()          # one pipeline run / model switch at a time
        self.backend = None
        self.model_name: str | None = None
        self.family: families.Family | None = None      # the loaded model's family
        self.state, self.error = "starting", ""         # starting | loading | ready | idle | error
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
        repos = parse_repos(getattr(args, "hub_repo", ""))
        self.hub = Hub(repos, self.models_dir, loaded=lambda: self.model_name, catalog=self._catalog) if repos else None
        self._load_dataset()

    def _base_config(self) -> AppConfig:
        """Default CYWS AppConfig, with the CLI's 'balanced' preset applied when the CLI defines presets."""
        return families.get("cyws").base_config()

    def active_family(self) -> "families.Family":
        """The loaded model's family; CYWS while nothing is loaded."""
        return self.family or families.get("cyws")

    def _base_for(self, fam: "families.Family"):
        return self.base_cfg if fam.name == "cyws" else fam.base_config()

    def presets(self) -> dict:
        return self.active_family().presets()

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
        """Models this demo knows: the built-in ones, plus any Hugging Face lists."""
        out = OrderedDict((n, dict(m)) for n, m in MODELS.items())
        for n, e in (self.hub.models() if self.hub else {}).items():
            builtin = MODELS.get(n)
            out[n] = {"file": e["files"][0]["name"], "bytes": e["files"][0]["bytes"],
                      "label": builtin["label"] if builtin else f"{n} · {e.get('title') or n}",
                      "family": builtin["family"] if builtin else e.get("family", "cyws"),
                      "title": builtin["title"] if builtin else (e.get("title") or n)}
        return out

    def _on_board(self, m: dict) -> bool:
        path = self.models_dir / m["file"]
        return path.is_file() and ("bytes" not in m or path.stat().st_size == m["bytes"])

    def available_models(self) -> list[dict]:
        return [{"name": n, "label": m["label"], "title": m.get("title", n), "family": m.get("family", "cyws"),
                 "available": self._on_board(m), "loaded": n == self.model_name} for n, m in self._catalog().items()]

    def delete_model(self, name: str) -> None:
        """Delete a model's file (and any stopped download of it) from the models directory.
        The loaded model is refused: load another first, so the MLA always runs a file that
        is there."""
        catalog = self._catalog()
        if name not in catalog:
            raise CywsError(f"unknown model {name!r}")
        if self.hub is not None and self.hub.busy_with(name):
            raise CywsError(f"{name} is downloading; cancel the download first")
        with self.lock:                                  # waits out a load or a run in progress
            if name == self.model_name or (self.state == "loading" and self.load_target == name):
                raise CywsError(f"{name} is loaded on the MLA. Load another model or unload it before deleting it")
            path = (self.models_dir / catalog[name]["file"]).resolve()
            if path.parent != self.models_dir.resolve():
                raise CywsError(f"refusing to delete a file outside the models directory: {path}")
            part = path.with_name(path.name + ".part")
            if not path.exists() and not part.exists():
                raise CywsError(f"{name} is not on this board")
            path.unlink(missing_ok=True)
            part.unlink(missing_ok=True)
            if self.hub is not None:
                self.hub.forget(name)
        log(f"deleted model {name} ({path})")

    def load_model(self, name: str) -> None:
        catalog = self._catalog()
        if name not in catalog:
            raise CywsError(f"unknown model {name!r}")
        fam = families.get(catalog[name].get("family", "cyws"))
        path = self.models_dir / catalog[name]["file"]
        if not path.exists():
            title = catalog[name].get("title") or name
            err = CywsError(f"{title} is not on this board yet. Download it in Models.")
            if self.backend is None:            # nothing loaded: say so instead of "starting" forever
                self.state, self.error = "idle", str(err)
            raise err
        with self.lock:
            self.state, self.error = "loading", ""
            t0 = time.perf_counter()
            self.load_target, self.load_t0 = name, t0
            try:
                if self.backend is not None:          # close before building: one runner at a time
                    self.load_stage = "release"
                    self.backend.close()
                    self.backend, self.model_name, self.family = None, None, None
                self.load_stage = "load"
                self.backend = fam.open(path, self.args.backend, self.args.onnx)
                # warm-up: first inference pays one-time costs
                self.load_stage = "warm"
                fam.warm(self.backend)
                self.model_name, self.family, self.state = name, fam, "ready"
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
                    self.backend, self.model_name, self.family = None, None, None

    def unload_model(self) -> dict:
        """Free the MLA: close the loaded model (waits out a run or a load) and go idle."""
        with self.lock:
            name = self.model_name
            try:
                if self.backend is not None:
                    self.backend.close()
            finally:
                self.backend, self.model_name, self.family = None, None, None
                self.state, self.error = "idle", ""
                self.load_stage = ""
                self.mla_ms.clear()
        if name:
            log(f"model {name} unloaded")
        return self.status()

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
    def config_from(self, opts: dict, family: "families.Family | None" = None):
        """The run config for the page's options, checked against the family's option spec."""
        fam = family or self.active_family()
        return fam.config_from(self._base_for(fam), opts)

    def run(self, pid: str, opts: dict) -> dict:
        if self.state == "idle":
            raise NoModelError(NO_MODEL)
        item = self.pairs.get(pid) or self.uploads.get(pid)
        if item is None:
            raise CywsError(f"unknown pair {pid!r}")
        fam = self.active_family()
        cfg = self.config_from(opts, fam)
        pair, views = self._load(pid, item, cfg, fam)  # CPU only: decodes outside the MLA lock
        with self.lock:
            if self.state == "idle":
                raise NoModelError(NO_MODEL)
            if self.state != "ready" or self.backend is None:
                raise CywsError(f"model is not ready ({self.state}{': ' + self.error if self.error else ''})")
            if self.family is not fam:                     # the model changed meanwhile: redo for the new one
                fam = self.family
                cfg = self.config_from(opts, fam)
                pair, views = self._load(pid, item, cfg, fam)
            res = fam.process(self.backend, pair, cfg)
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
            "family": fam.name,
        }
        stats = self.backend.stats() if self.backend is not None else {}
        lat = (stats.get("latency") or {}).get("mean_ms")
        if lat:
            self.mla_ms.append(float(lat))
            self.mla_ms[:] = self.mla_ms[-50:]
        return json.loads(json.dumps(out, default=_json_default))

    def _load(self, pid: str, item, cfg, fam: "families.Family | None" = None):
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
        pair = (fam or families.get("cyws")).load(item, cfg)
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
        fam = self.active_family()
        return {"app": "cyws-demo", "state": self.state, "error": self.error, "model": self.model_name,
                "family": self.family.name if self.family else None,
                "capabilities": dict(fam.capabilities),
                "families": {n: f.describe() for n, f in families.FAMILIES.items()},
                "models": self.available_models(), "backend": self.args.backend, "runs": self.runs,
                "pairs": len(self.pairs), "uploads": list(self.uploads), "presets": fam.presets(),
                "defaults": fam.defaults(self._base_for(fam)),
                "latency": stats.get("latency"), "https_port": self.args.port if self.args.tls else None,
                "load": {"stage": self.load_stage, "target": self.load_target,
                         "elapsed_s": round(time.perf_counter() - self.load_t0, 2) if self.load_stage else 0.0,
                         "expected_s": self.last_load_s.get(self.load_target)}}


NO_MODEL = "No model is loaded. Load one in Models."


class NoModelError(CywsError):
    """A run while no model is loaded (the page answers 409)."""


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
        resp.headers["Cache-Control"] = "public, max-age=604800, immutable" if name.startswith(("assets/", "vendor/")) else "no-cache"
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
        except NoModelError as e:
            return fail(e, 409)
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
            return jsonify({"repo": None, "listed": True, "models": {}, "error": None,
                            "directory": str(demo.models_dir), "free_bytes": _free_bytes(demo.models_dir),
                            **models_storage(demo.models_dir)})
        demo.hub.models(wait=demo.hub._models is None)        # the first visit waits for the list
        return jsonify(demo.hub.describe())

    @app.post("/api/hub/<action>")
    def hub_action(action):
        name = str((request.get_json(silent=True) or {}).get("name", ""))
        if action == "delete":                           # works without the hub too
            try:
                demo.delete_model(name)
            except Exception as e:
                return fail(e)
            return jsonify(demo.hub.describe() if demo.hub else {"repo": None, "listed": True, "models": {},
                                                                 "error": None, **models_storage(demo.models_dir)})
        if demo.hub is None:
            return fail(CywsError("this demo has no model hub (started with --hub-repo '')"))
        fn = {"download": demo.hub.download, "redownload": demo.hub.redownload, "verify": demo.hub.verify,
              "cancel": demo.hub.cancel, "discard": demo.hub.discard}.get(action)
        if fn is None:
            return fail(KeyError(f"unknown action {action}"), 404)
        try:
            result = fn(name)
            out = demo.hub.describe()
            if action == "verify":
                out["verified"] = {"name": name, "ok": bool(result)}
            return jsonify(out)
        except Exception as e:
            return fail(e)

    @app.post("/api/model")
    @app.post("/api/models/load")
    def model():
        name = (request.get_json(silent=True) or {}).get("name", "")
        try:
            demo.load_model(name)
            return jsonify(demo.status())
        except Exception as e:
            return fail(e)

    @app.post("/api/models/unload")
    def unload():
        try:
            return jsonify(demo.unload_model())
        except Exception as e:
            log(traceback.format_exc())
            return fail(e, 500)

    return app


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8098)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--model", default="bf16", choices=list(MODELS))
    ap.add_argument("--models", default=str(default_models_dir()),
                    help="the models directory (default: CYWS_MODELS_DIR, .models-dir, /media/nvme/models/cyws, or models/)")
    ap.add_argument("--data", default=os.environ.get("CYWS_DATA") or str(ROOT / "data/testCYWS"))
    ap.add_argument("--cache", default=str(ROOT / "webapp/cache"))
    ap.add_argument("--uploads", default=str(ROOT / "webapp/uploads"))
    ap.add_argument("--backend", default="neat", choices=["neat", "onnx", "fake"],
                    help="neat on the board; onnx/fake for trying the page on a PC")
    ap.add_argument("--onnx", default=str(ROOT / "build/onnx/cyws_fused.onnx"))
    ap.add_argument("--hub-repo", default=",".join(HUB_REPOS),
                    help="Hugging Face repositories whose models.json lists downloadable models, "
                         "comma-separated, first wins a name clash ('' for none)")
    ap.add_argument("--list-hub", action="store_true", help="list the models on Hugging Face and exit")
    ap.add_argument("--fetch", nargs="+", metavar="NAME",
                    help="download these models from Hugging Face into --models and exit ('all' for every one)")
    ap.add_argument("--delete", nargs="+", metavar="NAME",
                    help="delete these models from --models and exit (not one the running demo has loaded)")
    ap.add_argument("--verify", nargs="+", metavar="NAME",
                    help="check these models in --models against their published SHA-256 and exit ('all')")
    ap.add_argument("--no-https", dest="https", action="store_false",
                    help="serve plain HTTP only (by default the port also speaks HTTPS when a certificate "
                         "exists; browsers only allow the webcam on HTTPS pages)")
    ap.add_argument("--cert", default=str(ROOT / "webapp/certs/cert.pem"))
    ap.add_argument("--key", default=str(ROOT / "webapp/certs/key.pem"))
    args = ap.parse_args()
    if args.delete:
        return delete_command(args)
    if args.list_hub or args.fetch or args.verify:
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
    repos = parse_repos(args.hub_repo)
    if not repos:
        print("no --hub-repo: there is nothing to list or fetch", file=sys.stderr)
        return 1
    models_dir = Path(args.models).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    hub = Hub(repos, models_dir, loaded=lambda: None, catalog=lambda: {})
    hub.quiet = True
    hub._refresh()
    listed = hub.describe()
    for r in listed["repos"]:
        if r["error"]:
            print(r["error"], file=sys.stderr)
    if not any(r["listed"] for r in listed["repos"]):
        return 1
    for sk in listed["skipped"]:
        print(f"  skipped {sk['name']} from {sk['repo']}: {sk['reason']}", file=sys.stderr)
    mb = lambda n: f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"
    if getattr(args, "list_hub", False):
        pages = ", ".join(r["page"] for r in listed["repos"] if r["listed"])
        print(f"models in {pages} (into {models_dir}):")
        for name, m in listed["models"].items():
            print(f"  {name:12s} {m['title']:18s} {m['family']:7s} {mb(m['bytes']):>8s}  {m['state'].replace('_', ' ')}")
        print(f"on this board: {listed['model_count']} model(s), {mb(listed['used_bytes'])}; "
              f"{mb(listed['free_bytes'])} free")
        return 0
    if getattr(args, "verify", None):
        here = [n for n, m in listed["models"].items() if m["on_disk_bytes"]]
        names, bad = (here if args.verify == ["all"] else args.verify), 0
        if not names:
            print(f"no model to check in {models_dir}")
        for name in names:
            try:
                ok = hub.verify(name)
            except CywsError as e:
                print(f"  {name}: {e}", file=sys.stderr)
                bad += 1
                continue
            print(f"  {name}: {'matches the published file' if ok else hub._failed[name]}", file=sys.stdout if ok else sys.stderr)
            bad += not ok
        return 1 if bad else 0
    names = list(listed["models"]) if args.fetch == ["all"] else args.fetch
    failed = 0
    for name in names:
        m = listed["models"].get(name)
        if m is None:
            print(f"  {name}: not in {', '.join(repos)} (there: {', '.join(listed['models'])})", file=sys.stderr)
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


def delete_command(args) -> int:
    """`--delete NAME…`: remove models from the models directory. A model the demo running
    on this board has loaded is refused (it is asked over HTTP on --port)."""
    models_dir = Path(args.models).resolve()
    files = {n: m["file"] for n, m in MODELS.items()}
    repos = parse_repos(args.hub_repo)
    if repos:                                            # names Hugging Face adds, when it answers
        hub = Hub(repos, models_dir, loaded=lambda: None, catalog=lambda: {})
        hub._refresh()
        files.update({n: e["files"][0]["name"] for n, e in (hub._models or {}).items()})
    loaded = None
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{args.port}/api/status")
        with urllib.request.urlopen(req, timeout=2) as r:
            st = json.loads(r.read(1 << 20))
            if st.get("app") == "cyws-demo":
                load = st.get("load") or {}       # target is the last load's; only counts while loading
                loaded = st.get("model") or (load.get("target") if load.get("stage") or st.get("state") == "loading"
                                             else None)
    except (OSError, ValueError):
        pass
    names = [n for n in files if (models_dir / files[n]).exists()] if args.delete == ["all"] else args.delete
    failed = 0
    for name in names:
        if name not in files:
            print(f"  {name}: no such model (known: {', '.join(files)})", file=sys.stderr)
            failed += 1
            continue
        path = models_dir / files[name]
        part = path.with_name(path.name + ".part")
        if name == loaded:
            print(f"  {name}: the running demo has it loaded; load another model, Unload it in Models, "
                  "or stop the demo first",
                  file=sys.stderr)
            failed += 1
            continue
        if not path.exists() and not part.exists():
            print(f"  {name}: not here ({path})")
            continue
        size = sum(p.stat().st_size for p in (path, part) if p.exists())
        path.unlink(missing_ok=True)
        part.unlink(missing_ok=True)
        print(f"  {name}: deleted {path} ({size / 1e6:.0f} MB)")
    return 1 if failed else 0


def _safe_load(demo: Demo, name: str) -> None:
    try:
        demo.load_model(name)
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())
