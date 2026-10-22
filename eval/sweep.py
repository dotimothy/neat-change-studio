#!/usr/bin/env python3
"""Accuracy/cost sweep of CywsApp configurations over the test dataset, on the host GPU.

Every pair is decoded once (per load-relevant config: registration keeps a
native-size before image when the aspects differ, a future reduced-resolution
decode changes the frame), then app.CywsApp runs each config on every pair with
a CachedBackend in front of the torch model, so configs that differ only in
post-processing never recompute inference.

Execution: --workers N spawns N CPU worker processes that run
the pipeline (decoded pairs shared as memory-mapped .npy files); their cache
misses go over a queue to one GPU server thread in the main process, which
batches requests from all workers. --workers 0 runs
everything in-process (simplest, best for timing). host_ms is the app's own
time per pair without decode and without waiting for inference (total - load -
infer_wait); every app timing key is also kept as t_<key>. With workers, cv2
runs single-threaded per worker (--cv2-threads) and workers share the CPU, so
use --workers 0 for representative timings.

Grid (defaults): levels {0}, {0,1}, {0,1,2}; tile {auto, 0.35, 0.25} as a
fraction of the working frame's short side (finer levels only); tiles {full,
gated} (finer levels only); thr {0.15, 0.2, 0.25, 0.3, 0.4, 0.5}; reg
{off, on (= register + verify, box masks), on-diff (register + verify, diff masks)}.
Pruning (unless --full-grid): stage 1 runs every structure at --coarse-thr,
stage 2 runs the remaining thresholds for the --refine-top best structures;
everything skipped is listed in sweep.md / sweep.log.

Metrics per config: pooled pixel IoU/F1/P/R (tp/fp/fn summed over pairs), mean
per-pair F1, pooled GT component recall, pooled box precision, fp_pairs (empty-GT
pairs with >= 1 box), boxes/pair, windows/pair, host ms/pair.

Outputs (in --out, default build/sweep): sweep.csv (one row per config, ranked),
sweep_pairs.csv (one row per config x pair), sweep.md (ranked table, per-pair
breakdown of the top configs, skipped configs), sweep.log.

Examples:
  # smoke: 2 configs, all pairs
  .venv/bin/python eval/sweep.py --levels 0 --thr 0.3 --reg off,on --full-grid --out build/sweep_smoke
  # the default pruned sweep
  .venv/bin/python eval/sweep.py --workers 16
  # everything, no pruning
  .venv/bin/python eval/sweep.py --full-grid --workers 16
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import multiprocessing as mp
import os
import pickle
import queue
import signal
import shutil
import sys
import tempfile
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Hashable, Iterable, Optional, Sequence

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT / "app", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import app as cyws_app  # noqa: E402
import backends  # noqa: E402
import dataset_loader  # noqa: E402
import post_process  # noqa: E402
from cached_backend import CachedBackend  # noqa: E402
from common import AppConfig, CywsError, Pair, PairItem  # noqa: E402

DEFAULT_DATASET = Path(os.environ.get("CYWS_DATA") or ROOT / "data" / "testCYWS")
DEFAULT_CACHE = ROOT / "build/infer_cache"
DEFAULT_ONNX = ROOT / "build/onnx/cyws_fused.onnx"

LEVEL_SETS = {"0": (0,), "0,1": (0, 1), "0,1,2": (0, 1, 2)}
REG_MODES = {
    # name -> AppConfig overrides (verify_change None = on iff register)
    "off": dict(register=False, mask_mode="box", verify_change=None),
    "on": dict(register=True, mask_mode="box", verify_change=None),
    "on-diff": dict(register=True, mask_mode="diff", verify_change=None),
}
CV2_SEED = 12345
DEFAULT_THR = (0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
DEFAULT_COARSE_THR = (0.2, 0.3, 0.4)

# Fields that only steer CywsApp.process(), never ingestion.load_pair(); every other
# AppConfig field (including ones added later, e.g. a decode resolution) keys the decoded-pair cache.
PROCESS_ONLY = {
    "levels", "tile", "overlap", "tiles", "gate_heat", "gate_diff_percentile", "resize", "max_in_flight",
    "thr", "topk", "min_box_side", "edge_margin", "huge_box_frac", "nms_iou", "containment", "merge",
    "max_boxes", "fuse_iou", "reg_method", "photometric", "verify_change", "verify_zncc",
    "verify_diff_ratio", "min_valid_frac", "mask_mode", "save_heat", "preprocess_threads",
}


# =============================================================================== configs

@dataclasses.dataclass(frozen=True)
class Spec:
    """One sweep point; tile is 'auto' or a fraction of the working frame's short side."""

    levels: str
    tile: str
    tiles: str
    thr: float
    reg: str

    @property
    def finer(self) -> bool:
        return LEVEL_SETS[self.levels] != (0,)

    @property
    def structure(self) -> str:
        """Everything but the threshold."""
        lv = "L" + self.levels.replace(",", "")
        if not self.finer:
            return f"{lv}-{self.reg}"
        tile = "auto" if self.tile == "auto" else f"t{float(self.tile):.2f}"
        return f"{lv}-{tile}-{self.tiles}-{self.reg}"

    @property
    def name(self) -> str:
        return f"{self.structure}-thr{self.thr:g}"

    def overrides(self) -> dict:
        return dict(levels=LEVEL_SETS[self.levels], tiles=self.tiles if self.finer else "full",
                    thr=float(self.thr), **REG_MODES[self.reg])


def make_grid(levels: Sequence[str], tiles_frac: Sequence[str], tiles: Sequence[str], thrs: Sequence[float],
              regs: Sequence[str]) -> list[Spec]:
    """Cartesian grid; tile/tiles collapse to one value when there is no finer level."""
    out: dict[str, Spec] = {}
    for lv in levels:
        fin = LEVEL_SETS[lv] != (0,)
        for tf in (tiles_frac if fin else ["auto"]):
            for tl in (tiles if fin else ["full"]):
                for reg in regs:
                    for thr in thrs:
                        s = Spec(lv, tf, tl, float(thr), reg)
                        out.setdefault(s.name, s)
    return list(out.values())


def base_config(sets: Sequence[str]) -> AppConfig:
    """AppConfig() defaults (so new fields such as work_long keep their default) plus --set overrides."""
    cfg = AppConfig()
    fields = {f.name: f for f in dataclasses.fields(AppConfig)}
    over: dict[str, Any] = {}
    for item in sets:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        if k not in fields:
            raise SystemExit(f"--set: AppConfig has no field {k!r}; fields: {sorted(fields)}")
        cur = getattr(cfg, k)
        if isinstance(cur, bool):
            val: Any = v.lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int) and v != "auto":
            val = int(v)
        elif isinstance(cur, float):
            val = float(v)
        elif isinstance(cur, tuple):
            val = tuple(int(x) for x in v.split(",") if x)
        elif v.lower() == "none":
            val = None
        else:
            val = v
        over[k] = val
    return dataclasses.replace(cfg, **over)


def make_cfg(base: AppConfig, spec: Spec, h: int, w: int) -> AppConfig:
    over = spec.overrides()
    if spec.finer and spec.tile != "auto":
        # The decoded after image is the working frame (AppConfig.work_long, when present, is
        # applied at decode time), so the fraction is of the decoded short side.
        over["tile"] = max(64, int(round(float(spec.tile) * min(h, w))))
    else:
        over["tile"] = "auto"
    return dataclasses.replace(base, **over)


def load_key(cfg: AppConfig) -> tuple:
    return tuple((f.name, repr(getattr(cfg, f.name))) for f in dataclasses.fields(cfg) if f.name not in PROCESS_ONLY)


# =============================================================================== decoded pairs

class PairStore:
    """Decoded pairs per load key; arrays are made read-only so no config can corrupt them."""

    def __init__(self, items: Sequence[PairItem]) -> None:
        self.items = list(items)
        self._pairs: dict[tuple, list[Optional[Pair]]] = {}
        self.load_ms: dict[str, float] = {}

    @staticmethod
    def _freeze(pair: Pair) -> Pair:
        for a in (pair.before, pair.after, pair.gt_mask):
            if isinstance(a, np.ndarray):
                a.flags.writeable = False
        return pair

    def _reuse(self, key: tuple, idx: int) -> Optional[Pair]:
        """A pair decoded for a key that differs only in `register` is identical unless before_native."""
        k = dict(key)
        for other, pairs in self._pairs.items():
            o = dict(other)
            if pairs[idx] is not None and {a: b for a, b in o.items() if a != "register"} == \
                    {a: b for a, b in k.items() if a != "register"}:
                if not pairs[idx].meta.get("before_native", False):
                    return pairs[idx]
        return None

    def get(self, cfg: AppConfig, idx: int) -> Pair:
        key = load_key(cfg)
        pairs = self._pairs.setdefault(key, [None] * len(self.items))
        if pairs[idx] is None:
            pair = self._reuse(key, idx)
            if pair is None:
                t0 = time.perf_counter()
                pair = cyws_app.CywsApp(None, cfg).load(self.items[idx])
                self.load_ms[self.items[idx].pair_id] = (time.perf_counter() - t0) * 1e3
                pair = self._freeze(pair)
            pairs[idx] = pair
        return pairs[idx]

    def dump(self, root: Path) -> None:
        """Write every decoded pair as .npy files, for spawned workers to memory-map (shared, read-only)."""
        root.mkdir(parents=True, exist_ok=True)
        index: dict[tuple, list[Optional[str]]] = {}
        written: dict[int, str] = {}
        for key, pairs in self._pairs.items():
            names: list[Optional[str]] = []
            for p in pairs:
                if p is None:
                    names.append(None)
                    continue
                sub = written.get(id(p))
                if sub is None:
                    sub = written[id(p)] = f"p{len(written):04d}"
                    d = root / sub
                    d.mkdir()
                    np.save(d / "before.npy", p.before)
                    np.save(d / "after.npy", p.after)
                    if p.gt_mask is not None:
                        np.save(d / "gt.npy", p.gt_mask)
                    (d / "meta.pkl").write_bytes(pickle.dumps((p.pair_id, p.meta)))
                names.append(sub)
            index[key] = names
        (root / "index.pkl").write_bytes(pickle.dumps(index))

    @classmethod
    def from_dump(cls, items: Sequence[PairItem], root: Path) -> "PairStore":
        store = cls(items)
        loaded: dict[str, Pair] = {}

        def load(sub: str) -> Pair:
            if sub not in loaded:
                d = root / sub
                pid, meta = pickle.loads((d / "meta.pkl").read_bytes())
                mm = lambda f: np.load(d / f, mmap_mode="r").view(np.ndarray)  # noqa: E731 - read-only view
                loaded[sub] = Pair(pid, mm("before.npy"), mm("after.npy"),
                                   mm("gt.npy") if (d / "gt.npy").exists() else None, meta)
            return loaded[sub]

        for key, names in pickle.loads((root / "index.pkl").read_bytes()).items():
            store._pairs[key] = [None if n is None else load(n) for n in names]
        return store

    def preload(self, cfgs: Iterable[AppConfig], threads: int = 8, log: Callable[[str], None] = print) -> None:
        """Decode every pair for every distinct load key (in threads: PIL/cv2 release the GIL)."""
        seen: dict[tuple, AppConfig] = {}
        for c in cfgs:
            seen.setdefault(load_key(c), c)
        # register=True first: its decode can serve register=False too (see _reuse).
        order = sorted(seen.values(), key=lambda c: not getattr(c, "register", False))
        for cfg in order:
            t0 = time.perf_counter()
            failed: list[str] = []

            def one(i: int) -> None:
                try:
                    self.get(cfg, i)
                except CywsError as e:
                    failed.append(f"{self.items[i].pair_id}: {e}")

            with ThreadPoolExecutor(max(1, threads)) as ex:
                list(ex.map(one, range(len(self.items))))
            log(f"decoded {len(self.items)} pairs (register={getattr(cfg, 'register', None)}) in "
                f"{time.perf_counter() - t0:.1f} s" + (f"; failed: {failed}" if failed else ""))


# =============================================================================== backends

def cache_namespace(kind: str, ckpt: Path, onnx: Path) -> str:
    if kind in ("torch", "torch-bf16"):
        from torch_backend import cache_tag_for  # noqa: PLC0415 - no torch import at module level
        return cache_tag_for(ckpt, kind == "torch-bf16")
    if kind == "onnx":
        return f"onnx-{onnx.stem}"
    return "fake"


def make_inner(kind: str, ckpt: Path, onnx: Path, batch: int) -> backends.Backend:
    if kind in ("torch", "torch-bf16"):
        from torch_backend import TorchBackend  # noqa: PLC0415
        return TorchBackend(ckpt, bf16=kind == "torch-bf16", batch=batch)
    if kind == "onnx":
        return backends.OnnxBackend(onnx)
    if kind == "fake":
        return backends.FakeBackend()
    raise SystemExit(f"unknown backend {kind!r}")


class RemoteBackend(backends.Backend):
    """Worker side of the GPU server: requests go over a shared queue, results come back in order."""

    name = "remote"

    def __init__(self, wid: int, req_q: Any, resp_q: Any, max_in_flight: int = 16, timeout: float = 1800.0) -> None:
        super().__init__(max_in_flight)
        self.wid, self.req_q, self.resp_q, self.timeout = wid, req_q, resp_q, timeout
        self._seq = 0
        self._sent: deque[tuple[int, Hashable]] = deque()

    def _submit(self, tag: Hashable, before: np.ndarray, after: np.ndarray) -> None:
        self._seq += 1
        self.req_q.put((self.wid, self._seq, before, after))
        self._sent.append((self._seq, tag))

    def _collect(self) -> tuple[Hashable, np.ndarray]:
        seq, tag = self._sent.popleft()
        try:
            got, out, err = self.resp_q.get(timeout=self.timeout)
        except queue.Empty:
            raise CywsError(f"GPU server did not answer within {self.timeout:.0f} s") from None
        if err:
            raise CywsError(f"GPU server: {err}")
        if got != seq:
            raise CywsError(f"GPU server answered request {got}, expected {seq}")
        return tag, np.asarray(out, np.float32)

    def _abort(self, exc: BaseException) -> None:
        for _ in range(len(self._sent)):  # keep the response stream in sync for the next stream
            try:
                self.resp_q.get(timeout=self.timeout)
            except queue.Empty:
                break
        self._sent.clear()


def gpu_server(be: backends.Backend, req_q: Any, resp_qs: Sequence[Any], stop: threading.Event,
               batch: int, counter: dict) -> None:
    """Batch requests from all workers through one backend; float16 replies (what the cache stores)."""
    while not stop.is_set():
        try:
            reqs = [req_q.get(timeout=0.2)]
        except queue.Empty:
            continue
        while len(reqs) < batch:
            try:
                reqs.append(req_q.get_nowait())
            except queue.Empty:
                break
        try:
            outs = dict(be.infer_stream((i, r[2], r[3]) for i, r in enumerate(reqs)))
            for i, (wid, seq, _, _) in enumerate(reqs):
                resp_qs[wid].put((seq, outs[i].astype(np.float16), None))
            counter["batches"] = counter.get("batches", 0) + 1
            counter["inferences"] = counter.get("inferences", 0) + len(reqs)
        except Exception as e:  # noqa: BLE001 - report to the requesting workers
            msg = f"{type(e).__name__}: {e}"
            for wid, seq, _, _ in reqs:
                resp_qs[wid].put((seq, None, msg))


# =============================================================================== one task

def host_ms(t: dict) -> float:
    """App time per pair without decode and without waiting for inference.

    Uses the app's inference-wait key ("infer_wait", or "infer" in older apps).
    """
    wait = t.get("infer_wait", t.get("infer", 0.0))
    return float(t.get("total", 0.0)) - float(t.get("load", 0.0)) - float(wait)


def run_one(be: backends.Backend, store: PairStore, base: AppConfig, spec: Spec, idx: int) -> dict:
    """Run one config on one pair; return a flat row of metrics and timings."""
    item = store.items[idx]
    row: dict[str, Any] = {"config": spec.name, "pair_id": item.pair_id}
    try:
        cfg0 = dataclasses.replace(base, **spec.overrides())
        pair = store.get(cfg0, idx)
        h, w = pair.after.shape[:2]
        cfg = make_cfg(base, spec, h, w)
        # registration.register draws from cv2's global RNG (USAC/RANSAC sampling), so without a
        # reseed its homography depends on what ran before in the process: results would drift with
        # task scheduling and every registered crop would miss the inference cache.
        cv2.setRNGSeed(CV2_SEED)
        res = cyws_app.CywsApp(be, cfg).process(pair)
    except CywsError as e:
        row.update(status="failed", error=str(e))
        return row
    t = res.timings_ms
    row.update(status="ok", error="", height=h, width=w,
               tile_px=cfg.tile, windows=len(res.windows),
               windows_planned=res.meta.get("windows_planned"),
               n_boxes=len(res.detections),
               host_ms=round(host_ms(t), 2))
    # Every stage the app reports, whatever its timing keys are (they evolve with the app).
    row.update({f"t_{k}": round(float(v), 2) for k, v in t.items() if k not in ("load", "total")})
    if res.registration is not None:
        row.update(reg_ok=bool(res.registration.ok), reg_inliers=int(res.registration.inliers))
    if pair.gt_mask is not None:
        m = post_process.metrics(res.mask, pair.gt_mask, res.detections)
        row.update(gt_empty=(m["tp"] + m["fn"]) == 0, tp=m["tp"], fp=m["fp"], fn=m["fn"],
                   iou=round(m["iou"], 4), f1=round(m["f1"], 4),
                   gt_components=m["gt_components"], gt_components_hit=m["gt_components_hit"],
                   boxes_on_gt=int(round(m["box_precision"] * m["num_boxes"])))
    return row


# =============================================================================== aggregation

def aggregate(spec: Spec, rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]
    gt = [r for r in ok if "tp" in r]
    tp = sum(r["tp"] for r in gt)
    fp = sum(r["fp"] for r in gt)
    fn = sum(r["fn"] for r in gt)
    div = lambda a, b, d=0.0: (a / b) if b else d  # noqa: E731
    p, rc = div(tp, tp + fp), div(tp, tp + fn)
    comps = sum(r["gt_components"] for r in gt)
    nb = sum(r["n_boxes"] for r in gt)
    mean = lambda k: round(float(np.mean([r[k] for r in ok])), 2) if ok else None  # noqa: E731
    return {
        "config": spec.name, "structure": spec.structure, "levels": spec.levels,
        "tile": spec.tile if spec.finer else "-", "tiles": spec.tiles if spec.finer else "-",
        "thr": spec.thr, "reg": spec.reg,
        "pairs": len(rows), "failed": len(rows) - len(ok),
        "f1": round(div(2 * tp, 2 * tp + fp + fn), 4), "iou": round(div(tp, tp + fp + fn), 4),
        "precision": round(p, 4), "recall": round(rc, 4),
        "mean_f1": round(float(np.mean([r["f1"] for r in gt])), 4) if gt else None,
        "comp_recall": round(div(sum(r["gt_components_hit"] for r in gt), comps), 4),
        "comp_hit": sum(r["gt_components_hit"] for r in gt), "comp_total": comps,
        "box_precision": round(div(sum(r["boxes_on_gt"] for r in gt), nb), 4),
        "fp_pairs": sum(1 for r in gt if r["gt_empty"] and r["n_boxes"] > 0),
        "empty_pairs": sum(1 for r in gt if r["gt_empty"]),
        "boxes_per_pair": mean("n_boxes"), "windows_per_pair": mean("windows"),
        "host_ms_per_pair": mean("host_ms"),
        **{f"{k[2:]}_ms": mean(k) for k in dict.fromkeys(k for r in ok for k in r if k.startswith("t_"))},
    }


OBJECTIVES: dict[str, Callable[[dict], tuple]] = {
    "f1": lambda a: (a["f1"], -a["fp_pairs"], a["comp_recall"]),
    "iou": lambda a: (a["iou"], -a["fp_pairs"], a["comp_recall"]),
    "mean_f1": lambda a: (a["mean_f1"] or 0.0, a["f1"]),
    # pooled F1 with a 0.02 penalty per no-change pair that gets boxes
    "f1_fp": lambda a: (a["f1"] - 0.02 * a["fp_pairs"], a["comp_recall"]),
}


# =============================================================================== runners

class SerialRunner:
    def __init__(self, be: backends.Backend, store: PairStore, base: AppConfig) -> None:
        self.be, self.store, self.base = be, store, base

    def run(self, tasks: list[tuple[Spec, int]], progress: Callable[[int, int], None]) -> list[dict]:
        out = []
        for i, (spec, idx) in enumerate(tasks, 1):
            out.append(run_one(self.be, self.store, self.base, spec, idx))
            progress(i, len(tasks))
        return out

    def cache_counts(self) -> tuple[int, int]:
        return self.be.hits, self.be.misses

    def stats(self) -> dict:
        s = self.be.stats()
        return {"cache_hits": s.get("hits"), "cache_misses": s.get("misses"), "inner": s.get("inner")}

    def close(self, abort: bool = False) -> None:
        self.be.close()


def _worker_main(wid: int, task_q: Any, result_q: Any, req_q: Any, resp_q: Any, items: list[PairItem],
                 pairs_dir: str, base: AppConfig, cache_dir: str, namespace: str, cv2_threads: int) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the parent handles Ctrl-C and terminates us
    import cv2  # noqa: PLC0415
    cv2.setNumThreads(int(cv2_threads))
    store = PairStore.from_dump(items, Path(pairs_dir))  # pairs the worker lacks are decoded on demand
    be = CachedBackend(RemoteBackend(wid, req_q, resp_q), cache_dir, namespace=namespace)
    while True:
        task = task_q.get()
        if task is None:
            break
        tid, spec, idx = task
        try:
            row = run_one(be, store, base, spec, idx)
        except BaseException:  # noqa: BLE001 - a bug: report it, the parent aborts
            result_q.put((tid, None, traceback.format_exc(), wid, (be.hits, be.misses)))
            continue
        result_q.put((tid, row, None, wid, (be.hits, be.misses)))
    be.close()


class ParallelRunner:
    """N spawned CPU workers + one GPU server thread in this (parent) process.

    Workers are spawned, not forked: a fork after cv2 / torch started thread pools
    in the parent deadlocks the child. The decoded pairs reach them as .npy files
    they memory-map read-only, so the 28 pairs stay in RAM once (page cache).
    """

    def __init__(self, n: int, store: PairStore, base: AppConfig, cache_dir: Path, namespace: str,
                 make_backend: Callable[[], backends.Backend], batch: int, cv2_threads: int,
                 scratch: Path) -> None:
        self._tmp = Path(tempfile.mkdtemp(prefix=".pairs-", dir=scratch))
        store.dump(self._tmp)
        ctx = mp.get_context("spawn")
        self.task_q, self.result_q, self.req_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
        self.resp_qs = [ctx.Queue() for _ in range(n)]
        self.procs = [ctx.Process(target=_worker_main, daemon=True,
                                  args=(i, self.task_q, self.result_q, self.req_q, self.resp_qs[i], store.items,
                                        str(self._tmp), base, str(cache_dir), namespace, cv2_threads))
                      for i in range(n)]
        for p in self.procs:
            p.start()
        self.be = make_backend()
        self.stop = threading.Event()
        self.counter: dict = {}
        self._cache_hm: dict[int, tuple[int, int]] = {}
        self.server = threading.Thread(target=gpu_server, daemon=True,
                                       args=(self.be, self.req_q, self.resp_qs, self.stop, batch, self.counter))
        self.server.start()

    def run(self, tasks: list[tuple[Spec, int]], progress: Callable[[int, int], None]) -> list[dict]:
        for tid, (spec, idx) in enumerate(tasks):
            self.task_q.put((tid, spec, idx))
        out: list[Optional[dict]] = [None] * len(tasks)
        done = 0
        while done < len(tasks):
            try:
                tid, row, err, wid, hm = self.result_q.get(timeout=5.0)
            except queue.Empty:
                dead = [p.pid for p in self.procs if not p.is_alive()]
                if dead:
                    raise RuntimeError(f"sweep worker(s) {dead} died")
                continue
            if err:
                raise RuntimeError(f"task {tasks[tid][0].name} / {tasks[tid][1]} raised:\n{err}")
            self._cache_hm[wid] = hm
            out[tid] = row
            done += 1
            progress(done, len(tasks))
        return [r for r in out if r is not None]

    def cache_counts(self) -> tuple[int, int]:
        return sum(h for h, _ in self._cache_hm.values()), sum(m for _, m in self._cache_hm.values())

    def stats(self) -> dict:
        h, m = self.cache_counts()
        return {"cache_hits": h, "cache_misses": m, "server": dict(self.counter), "backend": self.be.stats()}

    def close(self, abort: bool = False) -> None:
        """Stop workers (after their queued tasks, or at once with abort) and the GPU server."""
        if abort:
            for p in self.procs:
                p.terminate()
        else:
            for _ in self.procs:
                self.task_q.put(None)
        for p in self.procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
        self.stop.set()
        self.server.join(timeout=5)
        self.be.close()
        shutil.rmtree(self._tmp, ignore_errors=True)


# =============================================================================== reporting

def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for r in rows:
        fields.extend(k for k in r if k not in fields)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.0f}"
    return str(v)


def write_md(path: Path, ranked: list[dict], pair_rows: dict[str, list[dict]], skipped: list[str],
             header: dict, top: int) -> None:
    cols = [("#", None), ("config", "config"), ("F1", "f1"), ("IoU", "iou"), ("P", "precision"),
            ("R", "recall"), ("mean F1", "mean_f1"), ("comp R", "comp_recall"), ("box P", "box_precision"),
            ("fp pairs", "fp_pairs"), ("boxes/pair", "boxes_per_pair"), ("win/pair", "windows_per_pair"),
            ("host ms/pair", "host_ms_per_pair"), ("failed", "failed")]
    L = ["# CYWS configuration sweep", ""]
    for k, v in header.items():
        L.append(f"- **{k}**: {v}")
    L += ["", f"## Ranked configs ({len(ranked)})", "",
          "| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)]
    for i, a in enumerate(ranked, 1):
        L.append("| " + " | ".join(str(i) if k is None else _fmt(a[k]) for _, k in cols) + " |")
    best = ranked[:top]
    if best:
        L += ["", f"## Per-pair breakdown of the top {len(best)}", "",
              "Cell: GT components hit/total, boxes, pixel F1 (no-change pairs: boxes only, `!` = false alarm).", "",
              "| pair | size | " + " | ".join(f"#{i}" for i in range(1, len(best) + 1)) + " |",
              "|---|---|" + "---|" * len(best)]
        legend = [f"#{i}: `{a['config']}`" for i, a in enumerate(best, 1)]
        by_cfg = [{r["pair_id"]: r for r in pair_rows[a["config"]]} for a in best]
        for pid in sorted(by_cfg[0]):
            r0 = by_cfg[0][pid]
            size = f"{r0.get('width', '?')}x{r0.get('height', '?')}"
            cells = []
            for d in by_cfg:
                r = d.get(pid, {})
                if r.get("status") != "ok":
                    cells.append("failed")
                elif "tp" not in r:
                    cells.append(f"{r['n_boxes']} boxes")
                elif r["gt_empty"]:
                    cells.append(f"{r['n_boxes']} boxes" + (" !" if r["n_boxes"] else ""))
                else:
                    cells.append(f"{r['gt_components_hit']}/{r['gt_components']}, {r['n_boxes']} b, F1 {r['f1']:.2f}")
            L.append(f"| {pid} | {size} | " + " | ".join(cells) + " |")
        L += [""] + [f"- {s}" for s in legend]
    L += ["", f"## Skipped by pruning ({len(skipped)})", ""]
    L += [f"- {s}" for s in skipped] if skipped else ["none"]
    path.write_text("\n".join(L) + "\n")


# =============================================================================== main

def parse_list(s: str, conv: Callable[[str], Any] = str, sep: str = ",") -> list:
    return [conv(x.strip()) for x in s.split(sep) if x.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--out", default=str(ROOT / "build/sweep"))
    ap.add_argument("--cache", default=str(DEFAULT_CACHE), help="inference cache root (shared across sweeps)")
    ap.add_argument("--backend", default="torch", choices=["torch", "torch-bf16", "onnx", "fake"])
    ap.add_argument("--ckpt", default=str(ROOT / "build/ckpt/pretrained-resnet50-3x-coam-scSE-affine.ckpt"))
    ap.add_argument("--onnx", default=str(DEFAULT_ONNX))
    ap.add_argument("--batch", type=int, default=16, help="GPU batch")
    ap.add_argument("--workers", type=int, default=min(16, max(1, (os.cpu_count() or 2) - 2)),
                    help="CPU worker processes; 0 = run in-process")
    ap.add_argument("--cv2-threads", type=int, default=1, help="cv2 threads per worker (with --workers > 0)")
    ap.add_argument("--levels", default="0;0,1;0,1,2", help="';'-separated level sets from 0 | 0,1 | 0,1,2")
    ap.add_argument("--tile", default="auto,0.35,0.25", help="auto and/or fractions of the working short side")
    ap.add_argument("--tiles", default="full,gated")
    ap.add_argument("--thr", default=",".join(f"{t:g}" for t in DEFAULT_THR))
    ap.add_argument("--reg", default="off,on,on-diff", help=f"from {list(REG_MODES)}")
    ap.add_argument("--full-grid", action="store_true", help="no pruning: run every grid point")
    ap.add_argument("--coarse-thr", default=",".join(f"{t:g}" for t in DEFAULT_COARSE_THR))
    ap.add_argument("--refine-top", type=int, default=6, help="structures refined over all --thr in stage 2")
    ap.add_argument("--objective", default="f1", choices=list(OBJECTIVES))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override an AppConfig field for every config (repeatable)")
    ap.add_argument("--pairs", default="", help="comma-separated pair ids (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="first N pairs only")
    ap.add_argument("--top", type=int, default=3, help="configs in the per-pair breakdown")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_f = (out / "sweep.log").open("w")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    levels = parse_list(args.levels, sep=";")
    bad = [lv for lv in levels if lv not in LEVEL_SETS]
    regs = parse_list(args.reg)
    if bad or any(r not in REG_MODES for r in regs):
        raise SystemExit(f"bad --levels {bad} or --reg {regs}")
    thrs = parse_list(args.thr, float)
    grid = make_grid(levels, parse_list(args.tile), parse_list(args.tiles), thrs, regs)
    base = base_config(args.set)

    items, warnings = dataset_loader.discover_pairs(args.dataset)
    for wmsg in warnings:
        log(f"dataset warning: {wmsg}")
    if args.pairs:
        keep = set(parse_list(args.pairs))
        items = [it for it in items if it.pair_id in keep]
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit("no pairs selected")
    extras = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)
              if f.name in ("work_long", "preprocess_threads") or f.name not in AppConfig.__dataclass_fields__}
    log(f"{len(items)} pairs, grid of {len(grid)} configs, backend {args.backend}, workers {args.workers}; "
        f"work_long={getattr(base, 'work_long', 'n/a (field absent)')}")

    # Stage plan.
    if args.full_grid:
        stages = [("all", grid)]
        coarse_thr: list[float] = []
    else:
        coarse_thr = [t for t in parse_list(args.coarse_thr, float) if t in thrs] or thrs
        stages = [("coarse", [s for s in grid if s.thr in coarse_thr])]

    store = PairStore(items)
    t0 = time.perf_counter()
    store.preload([dataclasses.replace(base, **s.overrides()) for s in grid], log=log)
    log(f"decode done in {time.perf_counter() - t0:.1f} s")

    namespace = cache_namespace(args.backend, Path(args.ckpt), Path(args.onnx))
    mk = lambda: make_inner(args.backend, Path(args.ckpt), Path(args.onnx), args.batch)  # noqa: E731
    if args.workers > 0:
        runner: Any = ParallelRunner(args.workers, store, base, Path(args.cache), namespace, mk, args.batch,
                                     args.cv2_threads, out)
    else:
        runner = SerialRunner(CachedBackend(mk(), args.cache, namespace=namespace, batch=args.batch), store, base)

    rows_by_cfg: dict[str, list[dict]] = {}
    aggs: dict[str, dict] = {}
    specs = {s.name: s for s in grid}
    rank_key = OBJECTIVES[args.objective]
    skipped: list[str] = []

    def run_stage(name: str, stage_specs: list[Spec]) -> None:
        todo = [s for s in stage_specs if s.name not in aggs]
        if not todo:
            return
        # One config per inference signature first, so concurrent configs do not both miss the cache.
        primers: dict[str, Spec] = {}
        for s in sorted(todo, key=lambda s: s.tiles != "full"):
            primers.setdefault(f"{s.levels}|{s.tile}|{s.reg != 'off'}", s)
        ordered = list(primers.values()) + [s for s in todo if s not in primers.values()]
        tasks = [(s, i) for s in ordered for i in range(len(items))]
        t_stage = time.perf_counter()
        last = [0.0]

        def progress(done: int, total: int) -> None:
            now = time.perf_counter()
            if now - last[0] > 15 or done == total:
                last[0] = now
                el = now - t_stage
                log(f"  {name}: {done}/{total} tasks, {el:.0f} s elapsed, ~{el / done * (total - done):.0f} s left")

        log(f"stage {name}: {len(todo)} configs x {len(items)} pairs ({len(primers)} cache primers first)")
        rows = runner.run(tasks, progress)
        for r in rows:
            rows_by_cfg.setdefault(r["config"], []).append(r)
        for s in todo:
            aggs[s.name] = aggregate(s, rows_by_cfg[s.name])
        h, m = runner.cache_counts()
        log(f"stage {name} done in {time.perf_counter() - t_stage:.0f} s; inference cache so far: {h} hits, {m} misses")

    try:
        for name, st in stages:
            run_stage(name, st)
        if not args.full_grid:
            ranked_structs: list[str] = []
            for a in sorted(aggs.values(), key=rank_key, reverse=True):
                if a["structure"] not in ranked_structs:
                    ranked_structs.append(a["structure"])
            top_structs = ranked_structs[: args.refine_top]
            refine = [s for s in grid if s.structure in top_structs and s.name not in aggs]
            log(f"refine: top {len(top_structs)} structures by {args.objective}: {top_structs}")
            run_stage("refine", refine)
            skipped = [s.name for s in grid if s.name not in aggs]
            log(f"skipped {len(skipped)} grid points (structures outside the top {args.refine_top} at "
                f"thr not in {coarse_thr}): " + ", ".join(skipped))
        rstats = runner.stats()
    except BaseException:
        runner.close(abort=True)
        raise
    runner.close()

    ranked = sorted(aggs.values(), key=rank_key, reverse=True)
    write_csv(out / "sweep.csv", ranked)
    write_csv(out / "sweep_pairs.csv", [r for a in ranked for r in sorted(rows_by_cfg[a["config"]],
                                                                           key=lambda r: r["pair_id"])])
    base_diff = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)
                 if getattr(base, f.name) != getattr(AppConfig(), f.name)}
    header = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "dataset": f"{args.dataset} ({len(items)} pairs, "
                   f"{sum(1 for r in rows_by_cfg[ranked[0]['config']] if r.get('gt_empty'))} with empty GT)",
        "backend": f"{args.backend} (cache namespace {namespace})",
        "base config": f"AppConfig defaults{'' if not base_diff else ' + ' + json.dumps(base_diff, default=str)}; "
                       f"extra fields: {json.dumps(extras, default=str) if extras else 'none'}",
        "grid": f"levels {levels}; tile {parse_list(args.tile)} (fraction of working short side); "
                f"tiles {parse_list(args.tiles)}; thr {thrs}; reg {regs}",
        "pruning": "none (--full-grid)" if args.full_grid else
                   f"coarse thr {coarse_thr}, then all thr for the top {args.refine_top} structures",
        "ranking": f"{args.objective} (desc), ties: fewer fp_pairs, higher component recall",
        "metrics": "pooled = tp/fp/fn summed over pairs; comp R = GT components hit / total; "
                   "box P = boxes with >= 30% area on GT / boxes; fp pairs = empty-GT pairs with >= 1 box; "
                   "host ms = app time per pair minus decode and inference wait (app.TIMING_KEYS)"
                   + (f" ({args.workers} parallel workers, cv2 threads {args.cv2_threads})" if args.workers else ""),
        "inference": json.dumps(rstats, default=str)[:600],
    }
    write_md(out / "sweep.md", ranked, rows_by_cfg, skipped, header, args.top)
    log(f"wrote {out / 'sweep.csv'}, {out / 'sweep_pairs.csv'}, {out / 'sweep.md'}")
    for i, a in enumerate(ranked[: min(10, len(ranked))], 1):
        log(f"  #{i} {a['config']}: F1 {a['f1']:.3f} IoU {a['iou']:.3f} compR {a['comp_recall']:.3f} "
            f"boxP {a['box_precision']:.3f} fp_pairs {a['fp_pairs']} win {a['windows_per_pair']} "
            f"host {a['host_ms_per_pair']} ms")
    log_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
