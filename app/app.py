"""CYWS change-detection pipeline: one decoded pair in, boxes + mask + heat out.

`CywsApp.process` glues the modules together in the order the plan fixes:
align -> plan windows -> stream crops through the backend -> decode -> filter
-> merge -> verify -> fuse -> mask. Everything stays in the canonical (after
image) frame defined in common.py, so the only coordinate conversions are the
window <-> canonical ones done inside post_process/tiling.
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

import numpy as np

import ingestion
import post_process
import registration
import tiling
from common import (
    CH,
    AppConfig,
    CywsError,
    Detection,
    Pair,
    PairItem,
    RegResult,
    Result,
    Window,
)

# Per-pair timings (ms). Wall-clock stages that add up:
#   total = load + register + stream + postprocess
#   stream ~= preprocess_wait + infer_wait + decode   (+ < 1 ms loop overhead)
# preprocess_cpu is the summed crop time of the worker threads; it overlaps
# the accelerator and other workers, so it is CPU time, not part of the sum.
TIMING_KEYS = (
    "load", "register", "stream", "preprocess_cpu", "preprocess_wait", "infer_wait", "decode",
    "postprocess", "total",
)
_LABELS = ("appeared", "disappeared", "changed")


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def jsonable(obj: Any) -> Any:
    """Recursively convert numpy / Path / tuple values into JSON-safe types.

    Result.meta and the summary end up in JSON files; numpy scalars from the
    sibling modules would otherwise make json.dump fail at the very end of a run.
    """
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist() if obj.size <= 64 else f"<ndarray {obj.shape} {obj.dtype}>"
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    return str(obj)


def validate_config(cfg: AppConfig) -> None:
    """Reject configurations that would fail deep inside the pipeline."""
    if not cfg.levels:
        raise CywsError("levels must not be empty")
    if any(lv not in (0, 1, 2) for lv in cfg.levels) or len(set(cfg.levels)) != len(cfg.levels):
        raise CywsError(f"levels must be distinct values from 0, 1, 2; got {cfg.levels}")
    if cfg.tile != "auto" and not (
        (isinstance(cfg.tile, int) and not isinstance(cfg.tile, bool) and cfg.tile >= 64)
        or (isinstance(cfg.tile, float) and 0.0 < cfg.tile <= 1.0)
    ):
        raise CywsError(f"tile must be 'auto', an integer >= 64 (pixels) or a float in (0, 1] "
                        f"(fraction of the short side); got {cfg.tile!r}")
    if not 0.0 <= cfg.gate_heat <= 1.0:
        raise CywsError(f"gate_heat must be in [0, 1]; got {cfg.gate_heat}")
    if not 0.0 <= cfg.overlap < 0.95:
        raise CywsError(f"overlap must be in [0, 0.95); got {cfg.overlap}")
    if not 0.0 < cfg.thr <= 1.0:
        raise CywsError(f"thr must be in (0, 1]; got {cfg.thr}")
    if cfg.topk < 1 or cfg.max_boxes < 1:
        raise CywsError("topk and max_boxes must be >= 1")
    if cfg.max_in_flight < 1:
        raise CywsError("max_in_flight must be >= 1")
    if not isinstance(cfg.work_long, int) or isinstance(cfg.work_long, bool) or (
        cfg.work_long != 0 and cfg.work_long < 256
    ):
        raise CywsError(f"work_long must be 0 (native resolution) or an integer >= 256; got {cfg.work_long!r}")
    if not isinstance(cfg.preprocess_threads, int) or isinstance(cfg.preprocess_threads, bool) or not (
        0 <= cfg.preprocess_threads <= 64
    ):
        raise CywsError(f"preprocess_threads must be an integer in [0, 64]; got {cfg.preprocess_threads!r}")
    choices = {
        "tiles": ("full", "gated"),
        "resize": ("area", "bilinear"),
        "merge": ("nms", "wbf"),
        "reg_method": ("sift", "orb"),
        "mask_mode": ("box", "diff"),
    }
    for name, allowed in choices.items():
        if getattr(cfg, name) not in allowed:
            raise CywsError(f"{name} must be one of {allowed}; got {getattr(cfg, name)!r}")


class CywsApp:
    """Runs the full CYWS pipeline for one pair at a time on a given backend.

    The backend is owned by the caller (it holds the accelerator); the app never
    closes it.
    """

    def __init__(self, backend: Any, cfg: Optional[AppConfig] = None) -> None:
        self.backend = backend
        self.cfg = cfg if cfg is not None else AppConfig()
        validate_config(self.cfg)

    # -- loading ---------------------------------------------------------------

    def load(self, item: PairItem) -> Pair:
        """Decode a pair at the working resolution with the app's squash/registration policy.

        The load time is stored in pair.meta["load_ms"] so process() can report
        it next to the other stages.
        """
        t0 = time.perf_counter()
        pair = ingestion.load_pair(
            item, allow_squash=self.cfg.allow_squash, register=self.cfg.register, work_long=self.cfg.work_long
        )
        pair.meta["load_ms"] = _ms(t0)
        return pair

    # -- pipeline --------------------------------------------------------------

    def process(self, pair: Pair) -> Result:
        """Detect changes between pair.before and pair.after (canonical frame)."""
        cfg = self.cfg
        t_start = time.perf_counter()
        timings = {k: 0.0 for k in TIMING_KEYS}
        timings["load"] = float(pair.meta.get("load_ms", 0.0))
        after = pair.after
        if after.ndim != 3 or after.shape[2] != 3 or after.dtype != np.uint8:
            raise CywsError(f"{pair.pair_id}: after must be uint8 (H, W, 3); got {after.dtype} {after.shape}")
        H, W = after.shape[:2]
        meta: dict[str, Any] = {"backend": getattr(self.backend, "name", type(self.backend).__name__)}
        # Canonical (working) frame -> original after-image pixels; hand-built pairs are native.
        meta["scale"] = tuple(float(v) for v in pair.meta.get("scale", (1.0, 1.0)))
        meta["orig_size"] = tuple(int(v) for v in pair.meta.get("orig_size", (W, H)))
        meta["work_size"] = (int(W), int(H))

        # 1. Align the before image into the canonical frame.
        t0 = time.perf_counter()
        before_aligned, reg, valid = self._align(pair)
        timings["register"] = _ms(t0)

        # 2. Plan windows (in the working frame, so auto tile = min(H, W) // 2 of it);
        #    under registration, skip windows that are mostly warp border.
        planned = tiling.plan_windows(H, W, tile=cfg.tile, overlap=cfg.overlap, levels=tuple(cfg.levels))
        windows = list(planned)
        if cfg.register:
            windows = [w for w in planned if tiling.valid_fraction(w, valid) >= cfg.min_valid_frac]
            if not any(w.level == 0 for w in windows):
                # A bad warp must not leave the pair unprocessed: keep the coarse view.
                windows = [w for w in planned if w.level == 0] + [w for w in windows if w.level > 0]
                meta["valid_frac_fallback"] = True
        meta["windows_planned"] = len(planned)
        meta["windows_skipped_invalid"] = len(planned) - len(windows)

        canvas = tiling.HeatCanvas(H, W, scale=4)
        raw: dict[str, list[Detection]] = {"after": [], "before": []}
        level0 = [w for w in windows if w.level == 0]
        finer = [w for w in windows if w.level > 0]

        # 3. Stream windows. Gated mode needs the level-0 heat before choosing finer windows.
        if cfg.tiles == "gated" and level0 and finer:
            run_windows = list(level0)
            self._stream(pair.pair_id, before_aligned, after, level0, canvas, raw, timings)
            # Diff energy only means "change" once the images are registered;
            # unregistered parallax would light up everywhere.
            energy = (
                registration.diff_energy_map(after, before_aligned, valid, scale=4) if cfg.register else None
            )
            kept = tiling.gate_windows(
                finer, canvas, energy, gate_heat=cfg.gate_heat, diff_percentile=cfg.gate_diff_percentile
            )
            meta["windows_gated_out"] = len(finer) - len(kept)
            if kept:
                self._stream(pair.pair_id, before_aligned, after, kept, canvas, raw, timings)
            run_windows += kept
        else:
            if cfg.tiles == "gated" and finer and not level0:
                meta["gating_skipped"] = "no level-0 windows to gate on"
            run_windows = level0 + finer
            self._stream(pair.pair_id, before_aligned, after, run_windows, canvas, raw, timings)
        meta["windows_run"] = len(run_windows)
        meta["windows_by_level"] = {
            str(lv): sum(1 for w in run_windows if w.level == lv) for lv in sorted({w.level for w in run_windows})
        }
        meta["raw_detections"] = {b: len(d) for b, d in raw.items()}

        # 4. Clean up, merge, verify and fuse.
        merged: dict[str, list[Detection]] = {}
        for branch, dets in raw.items():
            if cfg.register:
                dets = post_process.drop_invalid(dets, valid, max_invalid_frac=0.5)
            merged[branch] = post_process.merge(dets, run_windows, cfg)
        if cfg.verify_enabled():
            merged, dropped = self._verify(merged, after, before_aligned, valid)
            meta["verify_dropped"] = dropped
        detections = post_process.fuse_branches(merged["after"], merged["before"], iou=cfg.fuse_iou)
        mask = post_process.make_mask(
            detections,
            (H, W),
            mode=cfg.mask_mode,
            after=after,
            before_aligned=before_aligned,
            valid_mask=valid,
        )
        mask = np.asarray(mask, dtype=bool)

        meta["counts"] = {lab: sum(1 for d in detections if d.label == lab) for lab in _LABELS}
        meta["pair"] = pair.meta
        process_ms = _ms(t_start)
        # Everything outside alignment and the stream (planning, gating, merge,
        # verify, fuse, mask) is post-processing; by subtraction the wall-clock
        # stages add up to the total exactly.
        timings["postprocess"] = max(0.0, process_ms - timings["register"] - timings["stream"])
        timings["total"] = process_ms + timings["load"]
        return Result(
            pair_id=pair.pair_id,
            detections=detections,
            mask=mask,
            heat=np.asarray(canvas.array, dtype=np.float32),
            windows=run_windows,
            registration=reg,
            timings_ms={k: round(v, 3) for k, v in timings.items()},
            before_aligned=before_aligned,
            meta=jsonable(meta),
        )

    # -- stages ----------------------------------------------------------------

    def _align(self, pair: Pair) -> tuple[np.ndarray, Optional[RegResult], np.ndarray]:
        """Return (before in canonical frame, registration result, valid mask)."""
        cfg = self.cfg
        after = pair.after
        H, W = after.shape[:2]
        before = pair.before
        if before.ndim != 3 or before.shape[2] != 3 or before.dtype != np.uint8:
            raise CywsError(f"{pair.pair_id}: before must be uint8 (H, W, 3); got {before.dtype} {before.shape}")
        reg: Optional[RegResult] = None
        if cfg.register:
            reg = registration.register(before, after, method=cfg.reg_method)
            aligned = registration.warp_before(before, reg, after)
            valid = np.asarray(reg.valid_mask, dtype=bool)
        else:
            # load_pair only keeps a native-size before under registration; resize
            # defensively so a hand-built Pair still works.
            aligned = before if before.shape == after.shape else ingestion.resize_to(before, W, H)
            valid = np.ones((H, W), dtype=bool)
        if cfg.photometric:
            aligned = registration.photometric_match(aligned, after, valid)
        return np.ascontiguousarray(aligned), reg, valid

    def _stream(
        self,
        pair_id: str,
        before: np.ndarray,
        after: np.ndarray,
        windows: Sequence[Window],
        canvas: tiling.HeatCanvas,
        raw: dict[str, list[Detection]],
        timings: dict[str, float],
    ) -> None:
        """Push every window through the backend and decode outputs as they arrive.

        Crops run in cfg.preprocess_threads worker threads (cv2 releases the
        GIL), at most threads + max_in_flight windows ahead of the backend, so
        the CPU crops the next windows while the accelerator works. Windows
        are handed to the backend strictly in the given order with tags
        (pair_id, window.id). Timings added: stream (wall time of this loop),
        preprocess_cpu (summed worker crop time), preprocess_wait (main thread
        blocked on a crop), infer_wait (main thread blocked in the backend:
        push, pull, or a synchronous backend's compute) and decode.
        """
        cfg = self.cfg
        by_id = {w.id: w for w in windows}
        expected: deque[tuple[str, int]] = deque()
        acc = {"cpu": 0.0, "wait": 0.0}
        n_threads = min(int(cfg.preprocess_threads), len(windows))
        lookahead = n_threads + int(getattr(self.backend, "max_in_flight", cfg.max_in_flight) or 1)

        def crop(win: Window) -> tuple[np.ndarray, np.ndarray, float]:
            t0 = time.perf_counter()
            tb, ta = ingestion.crop_pair(before, after, win, interp=cfg.resize)
            return tb, ta, _ms(t0)

        pool = ThreadPoolExecutor(max_workers=n_threads, thread_name_prefix="cyws-crop") if n_threads > 0 else None

        def emit(win: Window, tb: np.ndarray, ta: np.ndarray):
            tag = (pair_id, win.id)
            expected.append(tag)
            return tag, tb, ta

        def items() -> Iterator[tuple[tuple[str, int], np.ndarray, np.ndarray]]:
            if pool is None:
                for win in windows:
                    t0 = time.perf_counter()
                    tb, ta, cpu = crop(win)
                    acc["wait"] += _ms(t0)
                    acc["cpu"] += cpu
                    yield emit(win, tb, ta)
                return
            queued: deque[tuple[Window, Future]] = deque()
            todo = iter(windows)

            def top_up() -> None:
                while len(queued) < lookahead:
                    nxt = next(todo, None)
                    if nxt is None:
                        return
                    queued.append((nxt, pool.submit(crop, nxt)))

            top_up()
            while queued:
                win, fut = queued.popleft()
                top_up()  # keep the workers busy while this window is consumed
                t0 = time.perf_counter()
                tb, ta, cpu = fut.result()
                acc["wait"] += _ms(t0)
                acc["cpu"] += cpu
                yield emit(win, tb, ta)

        t_stream = time.perf_counter()
        in_backend_ms = 0.0
        decode_ms = 0.0
        n_out = 0
        stream = self.backend.infer_stream(items())
        try:
            while True:
                t0 = time.perf_counter()
                try:
                    tag, out = next(stream)
                except StopIteration:
                    in_backend_ms += _ms(t0)
                    break
                in_backend_ms += _ms(t0)
                t0 = time.perf_counter()
                want = expected.popleft() if expected else None
                if tuple(tag) != want:
                    raise RuntimeError(f"backend returned tag {tag!r}, expected {want!r} (FIFO order broken)")
                out = np.asarray(out, dtype=np.float32)
                if out.shape != (len(CH), 256, 256):
                    raise RuntimeError(f"backend output shape {out.shape}, expected {(len(CH), 256, 256)}")
                win = by_id[tag[1]]
                canvas.add(win, np.maximum(out[CH["heat_a"]], out[CH["heat_b"]]))
                for branch in ("after", "before"):
                    dets256 = post_process.decode_centernet(out, branch, thr=cfg.thr, topk=cfg.topk)
                    raw[branch].extend(post_process.filter_window(dets256, win, branch, cfg))
                n_out += 1
                decode_ms += _ms(t0)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
        stream_ms = _ms(t_stream)
        if n_out != len(windows):
            raise RuntimeError(f"backend returned {n_out} outputs for {len(windows)} windows")
        timings["stream"] += stream_ms
        timings["preprocess_cpu"] += acc["cpu"]
        timings["preprocess_wait"] += acc["wait"]
        # items() runs inside next(stream), so its crop waits are taken out here.
        timings["infer_wait"] += max(0.0, in_backend_ms - acc["wait"])
        timings["decode"] += decode_ms

    def _verify(
        self,
        merged: dict[str, list[Detection]],
        after: np.ndarray,
        before_aligned: np.ndarray,
        valid: np.ndarray,
    ) -> tuple[dict[str, list[Detection]], list[dict]]:
        """Drop boxes whose after and aligned-before patches still match (parallax FPs)."""
        cfg = self.cfg
        if not any(merged.values()):
            return merged, []
        global_median = registration.median_abs_diff(after, before_aligned, valid if valid.any() else None)
        kept: dict[str, list[Detection]] = {}
        dropped: list[dict] = []
        for branch, dets in merged.items():
            kept[branch] = []
            for det in dets:
                keep, info = registration.verify_detection(
                    det,
                    after,
                    before_aligned,
                    zncc_thr=cfg.verify_zncc,
                    diff_ratio=cfg.verify_diff_ratio,
                    global_median_diff=global_median,
                )
                if keep:
                    kept[branch].append(det)
                else:
                    dropped.append({**det.to_dict(), "verify": jsonable(info)})
        return kept, dropped


# -- dataset driver ----------------------------------------------------------------


def _metric_scalars(m: dict) -> dict[str, Any]:
    """Keep the flat scalar entries of a metrics dict (CSV-friendly)."""
    out: dict[str, Any] = {}
    for k, v in m.items():
        v = jsonable(v)
        if v is None or isinstance(v, (bool, int, float, str)):
            out[k] = v
    return out


def _mean_row(rows: list[dict]) -> dict[str, Any]:
    """Column-wise mean of numeric fields over rows that have them."""
    mean: dict[str, Any] = {"pair_id": "mean", "status": f"{len(rows)} ok"}
    keys: list[str] = []
    for r in rows:
        keys.extend(k for k in r if k not in keys)
    for k in keys:
        if k in ("pair_id", "status", "error"):
            continue
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)]
        if vals:
            mean[k] = round(float(np.mean(vals)), 4)
    return mean


def _pooled(rows: list[dict]) -> dict[str, float]:
    """Pixel metrics from summed tp/fp/fn: one big image instead of a per-pair average."""
    with_gt = [r for r in rows if all(isinstance(r.get(k), (int, float)) for k in ("tp", "fp", "fn"))]
    if not with_gt:
        return {}
    tp = float(sum(r["tp"] for r in with_gt))
    fp = float(sum(r["fp"] for r in with_gt))
    fn = float(sum(r["fn"] for r in with_gt))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "pairs": len(with_gt),
        "iou": round(tp / (tp + fp + fn), 4) if tp + fp + fn else 0.0,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for r in rows:
        fields.extend(k for k in r if k not in fields)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})


def run_dataset(
    app: CywsApp,
    items: Iterable[PairItem],
    out_dir: str | Path,
    *,
    log: Callable[[str], None] = print,
) -> dict:
    """Process every pair, write per-pair outputs plus summary.csv / metrics.json.

    A pair that fails validation (CywsError) is logged and recorded as failed so
    one bad photo does not abort a long board run; any other exception (a bug,
    or Ctrl-C) propagates, but summary.csv / metrics.json are still written
    for the pairs finished so far, with the aborting pair recorded as failed.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = list(items)
    rows: list[dict] = []
    ok_rows: list[dict] = []
    failed: list[dict] = []
    current: Optional[str] = None
    # Decode the next pair while this one is cropped, inferred and merged: the JPEG
    # decode is CPU work on other cores, so it hides behind the stream. t_load keeps
    # reporting the decode's own duration.
    prefetch = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cyws-prefetch")
    pending = prefetch.submit(app.load, items[0]) if items else None
    # Overlay / mask / JSON files are written in the background too (JPEG + PNG encode
    # is ~200 ms per 3 MP pair on the A65); at most one pair's write is outstanding.
    writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cyws-writer")
    writing: Optional[tuple[Future, dict, str]] = None

    def settle_write() -> None:
        nonlocal writing
        if writing is None:
            return
        fut, wrow, wprefix = writing
        writing = None
        try:
            wrow["files"] = ";".join(str(v) for v in fut.result().values())
        except CywsError as e:
            log(f"{wprefix}: FAILED: {e}")
            ok_rows.remove(wrow)
            for k in list(wrow):
                if k not in ("pair_id",):
                    del wrow[k]
            wrow.update(status="failed", error=str(e))
            failed.append(wrow)
    t_run = time.perf_counter()
    try:
        for i, item in enumerate(items, 1):
            prefix = f"[{i}/{len(items)}] {item.pair_id}"
            current = item.pair_id
            loading, pending = pending, (prefetch.submit(app.load, items[i]) if i < len(items) else None)
            try:
                pair = loading.result()
                result = app.process(pair)
            except CywsError as e:
                log(f"{prefix}: FAILED: {e}")
                fail_row = {"pair_id": item.pair_id, "status": "failed", "error": str(e)}
                rows.append(fail_row)
                failed.append(fail_row)
                continue
            counts = {lab: sum(1 for d in result.detections if d.label == lab) for lab in _LABELS}
            row: dict[str, Any] = {
                "pair_id": result.pair_id,
                "status": "ok",
                "error": "",
                "height": int(pair.after.shape[0]),
                "width": int(pair.after.shape[1]),
                "orig_height": int(result.meta.get("orig_size", (0, pair.after.shape[0]))[1]),
                "orig_width": int(result.meta.get("orig_size", (pair.after.shape[1], 0))[0]),
                "n_boxes": len(result.detections),
                **{f"n_{lab}": n for lab, n in counts.items()},
                "windows": len(result.windows),
            }
            reg = result.registration
            if reg is not None:
                row.update(reg_ok=bool(reg.ok), reg_method=reg.method, reg_inliers=int(reg.inliers),
                           reg_inlier_ratio=round(float(reg.inlier_ratio), 4))
            row.update({f"t_{k}_ms": round(float(v), 2) for k, v in result.timings_ms.items()})
            mask_meta = pair.meta.get("mask") or {}
            if "read_ms" in mask_meta:  # GT mask decode, already inside t_load (evaluation-only cost)
                row["mask_read_ms"] = round(float(mask_meta["read_ms"]), 2)
            if pair.gt_mask is not None:
                row.update(_metric_scalars(post_process.metrics(result.mask, pair.gt_mask, result.detections)))
            row["files"] = ""
            rows.append(row)
            ok_rows.append(row)
            settle_write()
            writing = (writer.submit(post_process.save_outputs, result, pair.after, out,
                                     save_heat=app.cfg.save_heat), row, prefix)
            t = result.timings_ms
            msg = (f"{prefix}: {row['n_boxes']} boxes (A{counts['appeared']}/D{counts['disappeared']}"
                   f"/C{counts['changed']}), {row['windows']} windows, {t['total']:.0f} ms "
                   f"[load {t['load']:.0f} reg {t['register']:.0f} stream {t['stream']:.0f} "
                   f"(prep cpu {t['preprocess_cpu']:.0f} wait {t['preprocess_wait']:.0f}, "
                   f"infer wait {t['infer_wait']:.0f}, decode {t['decode']:.0f}) post {t['postprocess']:.0f}]")
            if "f1" in row:
                msg += f", F1 {row['f1']:.3f}"
            if reg is not None:
                msg += f", reg {reg.method}{'' if reg.ok else ' (rejected)'} {reg.inliers} inliers"
            log(msg)
            current = None
        settle_write()
    except BaseException as e:
        aborted = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
        if current is not None:
            fail_row = {"pair_id": current, "status": "failed", "error": f"run aborted: {aborted}"}
            rows.append(fail_row)
            failed.append(fail_row)
        if pending is not None:
            pending.cancel()
        try:
            settle_write()
        except BaseException as w:  # noqa: BLE001 - keep the original exception
            log(f"could not finish writing the last pair's files: {type(w).__name__}: {w}")
        log(f"run aborted after {len(ok_rows)} ok pair(s): {aborted}; writing partial summary")
        try:
            _write_summary(app, items, out, rows, ok_rows, failed, aborted)
        except Exception as w:  # noqa: BLE001 - keep the original exception
            log(f"could not write the partial summary: {type(w).__name__}: {w}")
        raise
    finally:
        prefetch.shutdown(wait=True, cancel_futures=True)
        writer.shutdown(wait=True)
    if items:
        wall = time.perf_counter() - t_run
        log(f"wall time {wall:.1f} s for {len(items)} pairs ({wall * 1e3 / len(items):.0f} ms per pair, "
            "decode of the next pair overlapped)")
    return _write_summary(app, items, out, rows, ok_rows, failed, None)


def _write_summary(app: CywsApp, items: list[PairItem], out: Path, rows: list[dict],
                   ok_rows: list[dict], failed: list[dict], aborted: Optional[str]) -> dict:
    """Write summary.csv and metrics.json; called even when the run aborts."""
    mean = _mean_row([{k: v for k, v in r.items() if k != "files"} for r in ok_rows]) if ok_rows else {}
    csv_rows = [{k: v for k, v in r.items() if k != "files"} for r in rows]
    _write_csv(out / "summary.csv", csv_rows + ([mean] if mean else []))
    stats_fn = getattr(app.backend, "stats", None)
    try:
        backend_stats = jsonable(stats_fn()) if callable(stats_fn) else {}
    except Exception as e:  # noqa: BLE001 - never lose the summary over a stats failure
        backend_stats = {"error": f"{type(e).__name__}: {e}"}
    summary = {
        "n_pairs": len(items),
        "n_ok": len(ok_rows),
        "n_failed": len(failed),
        "aborted": aborted,
        "backend": getattr(app.backend, "name", type(app.backend).__name__),
        "backend_stats": backend_stats,
        "config": jsonable(asdict(app.cfg)),
        "pairs": jsonable(rows),
        "mean": jsonable(mean),
        "pooled": _pooled(ok_rows),
        "timing_keys": {
            "sum": "total = load + register + stream + postprocess",
            "stream": "stream ~= preprocess_wait + infer_wait + decode; preprocess_cpu is summed worker time",
        },
        "failed": jsonable(failed),
        "out_dir": str(out),
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary
