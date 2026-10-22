"""Inference backends: one interface over the DevKit (pyneat), host ONNX, host afe and a fake.

Every backend takes the model contract from common.py: two float32 (256, 256, 3)
HWC RGB /255 tensors (before, after) per inference, and returns the canonical
float32 (10, 256, 256) CHW output, however the runtime lays it out. The app
streams windows through ``infer_stream`` so the accelerator always has up to
``max_in_flight`` pairs queued while the host decodes earlier results.

onnxruntime exists only on the host, pyneat only on the board and afe only in
the Model Compiler container, so each is imported inside its backend.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Hashable, Iterable, Iterator, Optional, Sequence

import cv2
import numpy as np

import common
from common import BRANCH_SLICES, INPUT_NAMES, MODEL_SIZE, OUT_CHANNELS, CywsError

N_OUT = len(OUT_CHANNELS)
IN_SHAPE = (MODEL_SIZE, MODEL_SIZE, 3)
# Split output layout, in this order: heat (b, a), wh (b_w, b_h, a_w, a_h), off (b_x, b_y, a_x, a_y).
SPLIT_PARTS = (("heat", 2), ("wh", 4), ("off", 4))

Item = tuple[Hashable, np.ndarray, np.ndarray]


# --------------------------------------------------------------------------- layout helpers

def _as_chw(arr: np.ndarray, channels: int, what: str) -> np.ndarray:
    """One output tensor (with optional batch 1) -> (channels, 256, 256), possibly a view."""
    a = np.asarray(arr)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    s = MODEL_SIZE
    if a.shape == (channels, s, s):
        return a
    if a.shape == (s, s, channels):
        return a.transpose(2, 0, 1)
    raise CywsError(
        f"{what}: unexpected output shape {tuple(np.shape(arr))}; expected "
        f"({channels},{s},{s}) or ({s},{s},{channels}), optionally with a leading batch of 1"
    )


def to_canonical(outputs: Any) -> np.ndarray:
    """Model outputs in any supported layout -> canonical float32 (10, 256, 256).

    Accepts the fused tensor as (1,10,256,256), (10,256,256), (256,256,10) or
    (1,256,256,10), or the split list [heat (2ch), wh (4ch), off (4ch)] with each
    part in NCHW or HWC. The result is always a fresh C-contiguous array: runtime
    output buffers are pooled, and holding a view into one stalls the pipeline.
    """
    if isinstance(outputs, (list, tuple)):
        arrays = [np.asarray(o) for o in outputs]
    else:
        arrays = [np.asarray(outputs)]
    if len(arrays) == 1:
        chw = _as_chw(arrays[0], N_OUT, "fused output")
        return np.array(chw, dtype=np.float32, order="C", copy=True)
    if len(arrays) == len(SPLIT_PARTS):
        parts = [_as_chw(a, c, f"split output '{n}'") for a, (n, c) in zip(arrays, SPLIT_PARTS)]
        return np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32)
    raise CywsError(
        f"expected 1 fused output or 3 split outputs (heat, wh, off), got {len(arrays)} "
        f"with shapes {[tuple(a.shape) for a in arrays]}"
    )


def _order_split_by_names(arrays: list[np.ndarray], names: Sequence[str]) -> list[np.ndarray]:
    """Reorder 3 split outputs to (heat, wh, off) when their names say which is which.

    Runtimes may return outputs sorted by name or by memory slot rather than in
    graph order; wh and off both have 4 channels, so only names can tell them apart.
    Leaves the order unchanged when the names are missing or ambiguous.
    """
    if len(arrays) != 3 or len(names) != 3:
        return arrays
    keys = {"heat": ("heat", "hm"), "wh": ("wh", "size"), "off": ("off", "reg")}
    order: list[int] = []
    for part, _ in SPLIT_PARTS:
        hits = [i for i, n in enumerate(names) if n and any(k in n.lower() for k in keys[part])]
        if len(hits) != 1:
            return arrays
        order.append(hits[0])
    if sorted(order) != [0, 1, 2]:
        return arrays
    return [arrays[i] for i in order]


def _check_input(x: Any, what: str) -> np.ndarray:
    """Validate one model input and return it as C-contiguous float32 (256, 256, 3)."""
    a = np.asarray(x)
    if a.shape != IN_SHAPE:
        raise CywsError(f"{what} input must have shape {IN_SHAPE} (HWC), got {a.shape}")
    if not np.issubdtype(a.dtype, np.floating):
        raise CywsError(f"{what} input must be float (pixel / 255), got dtype {a.dtype}")
    return np.ascontiguousarray(a, dtype=np.float32)


_MODEL_SUFFIXES = (".tar.gz", ".tgz", ".tar", ".onnx", ".elf", ".sima")


def model_cache_tag(kind: str, path: str | Path) -> str:
    """'<kind>-<model file name without archive/model suffix>', e.g. neat-cyws_bf16_mpk.

    Used as a backend's ``cache_tag``: an inference cache keyed only by the input
    bytes must also know which model produced the output, so two different model
    files (FP32 vs bf16, bf16 vs w8a16) never share cache entries.
    """
    name = Path(path).name
    for suf in _MODEL_SUFFIXES:
        if name.lower().endswith(suf):
            name = name[: -len(suf)]
            break
    return f"{kind}-{name}"


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    v = np.asarray(values, np.float64)
    return {
        "mean_ms": round(float(v.mean()), 3),
        "p50_ms": round(float(np.percentile(v, 50)), 3),
        "p90_ms": round(float(np.percentile(v, 90)), 3),
        "max_ms": round(float(v.max()), 3),
    }


# --------------------------------------------------------------------------- base class

class Backend:
    """Base backend: FIFO streaming with a bounded number of outstanding pushes.

    Subclasses implement ``_submit(tag, before, after)`` (queue one pair),
    ``_collect() -> (tag, out)`` (oldest result, canonical) and ``_close()``.
    Synchronous runtimes compute in ``_submit`` and queue the result.

    Latency bookkeeping (``stats()``): for an asynchronous runtime one
    inference's latency runs from the start of its push to the moment its
    output is in host memory (the pull returned); ``collect_wait`` is the
    part of that the caller spent blocked in the pull. Synchronous runtimes
    override ``_latency_ms`` and report the blocking compute time itself.
    """

    name = "backend"

    def __init__(self, max_in_flight: int = 2) -> None:
        if int(max_in_flight) < 1:
            raise CywsError(f"max_in_flight must be >= 1, got {max_in_flight}")
        self.max_in_flight = int(max_in_flight)
        self._closed = False
        self._broken: Optional[str] = None
        self._stream_guard = threading.Lock()
        self._streaming = False
        self._latencies_ms: list[float] = []
        self._collect_wait_ms: list[float] = []
        self._max_seen_in_flight = 0

    # -- subclass hooks
    def _submit(self, tag: Hashable, before: np.ndarray, after: np.ndarray) -> None:
        raise NotImplementedError

    def _collect(self) -> tuple[Hashable, np.ndarray]:
        raise NotImplementedError

    def _close(self) -> None:
        pass

    def _abort(self, exc: BaseException) -> None:
        """Called when a submit/collect failed mid-stream; default: nothing to resync."""

    def _latency_ms(self, t_push: float, t_done: float) -> float:
        """Latency of the result just collected: push start -> output in host memory."""
        return (t_done - t_push) * 1e3

    # -- public API
    def infer_stream(self, items: Iterable[Item]) -> Iterator[tuple[Hashable, np.ndarray]]:
        """Yield (tag, canonical output) for each (tag, before_hwc, after_hwc), in input order.

        At most ``max_in_flight`` pairs are outstanding. Abandoning the iterator
        early drains what is still queued, so the next stream starts in sync.
        """
        self._check_usable()
        with self._stream_guard:
            if self._streaming:
                raise CywsError(f"{self.name} backend is already streaming; one stream at a time")
            self._streaming = True
        pending: deque[tuple[Hashable, float]] = deque()
        try:
            for item in items:
                try:
                    tag, before, after = item
                except (TypeError, ValueError):
                    raise CywsError("infer_stream items must be (tag, before_hwc, after_hwc)") from None
                b = _check_input(before, "before")
                a = _check_input(after, "after")
                while len(pending) >= self.max_in_flight:
                    yield self._pop(pending)
                t_push = time.perf_counter()
                self._call(self._submit, tag, b, a)
                pending.append((tag, t_push))
                self._max_seen_in_flight = max(self._max_seen_in_flight, len(pending))
            while pending:
                yield self._pop(pending)
        finally:
            # Early exit or a caller-side error (e.g. a bad input) leaves results queued:
            # drain them so they are not paired with the next stream's tags.
            if pending and self._broken is None:
                try:
                    for _ in range(len(pending)):
                        self._call(self._collect)
                except BaseException:  # noqa: BLE001 - already reported via _abort/_broken
                    pass
            with self._stream_guard:
                self._streaming = False

    def infer(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """One pair -> canonical (10, 256, 256) float32."""
        tag = object()
        for _, out in self.infer_stream([(tag, before, after)]):
            return out
        raise CywsError(f"{self.name} backend produced no output")

    def stats(self) -> dict:
        """Per-inference latency and counters.

        latency: push start -> output in host memory (asynchronous runtimes;
        includes queueing behind earlier pushes, and is an upper bound when the
        caller pulls late) or the blocking compute time (synchronous runtimes).
        collect_wait: time the caller was blocked waiting for each output.
        """
        return {
            "backend": self.name,
            "inferences": len(self._latencies_ms),
            "max_in_flight": self.max_in_flight,
            "max_seen_in_flight": self._max_seen_in_flight,
            "latency": _percentiles(self._latencies_ms),
            "collect_wait": _percentiles(self._collect_wait_ms),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close()

    def __enter__(self) -> "Backend":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- internals
    def _check_usable(self) -> None:
        if self._closed:
            raise CywsError(f"{self.name} backend is closed")
        if self._broken is not None:
            raise CywsError(f"{self.name} backend is unusable after an earlier failure: {self._broken}")

    def _call(self, fn: Any, *args: Any) -> Any:
        try:
            return fn(*args)
        except BaseException as exc:
            self._abort(exc)
            raise

    def _pop(self, pending: deque) -> tuple[Hashable, np.ndarray]:
        exp_tag, t_push = pending[0]
        t0 = time.perf_counter()
        tag, out = self._call(self._collect)
        t_done = time.perf_counter()
        pending.popleft()
        if tag is not exp_tag:
            self._broken = "result order mismatch"
            raise CywsError(f"{self.name}: result for tag {tag!r} arrived while expecting {exp_tag!r}")
        self._collect_wait_ms.append((t_done - t0) * 1e3)
        self._latencies_ms.append(self._latency_ms(t_push, t_done))
        return tag, out


class _SyncBackend(Backend):
    """Backends whose runtime call is blocking: compute on submit, queue the result.

    The compute happens inside the push, so push -> collect would only time
    the queue hand-off; the latency reported for each result is the time
    spent in ``_run`` instead.
    """

    def __init__(self, max_in_flight: int = 2) -> None:
        super().__init__(max_in_flight)
        self._results: deque[tuple[Hashable, np.ndarray, float]] = deque()
        self._last_compute_ms = 0.0

    def _run(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _submit(self, tag: Hashable, before: np.ndarray, after: np.ndarray) -> None:
        t0 = time.perf_counter()
        out = self._run(before, after)
        self._results.append((tag, out, (time.perf_counter() - t0) * 1e3))

    def _collect(self) -> tuple[Hashable, np.ndarray]:
        if not self._results:
            raise CywsError(f"{self.name}: collect without a pending submit")
        tag, out, self._last_compute_ms = self._results.popleft()
        return tag, out

    def _latency_ms(self, t_push: float, t_done: float) -> float:
        return self._last_compute_ms

    def _abort(self, exc: BaseException) -> None:
        self._results.clear()


# --------------------------------------------------------------------------- pyneat (DevKit)

def _enum_name(value: Any) -> str:
    return str(getattr(value, "name", value))


def _spec_shape(spec: Any) -> tuple[int, ...]:
    return tuple(int(s) for s in (getattr(spec, "shape", None) or ()))


def _validate_input_specs(specs: Sequence[Any]) -> list[str]:
    """Check the MPK takes two float32 (256, 256, 3) inputs; return notes for stats().

    The order cannot be checked here (the specs carry no names), so the eval
    A/B-swap test verifies before/after order on the board.
    """
    specs = list(specs)
    if len(specs) != len(INPUT_NAMES):
        raise CywsError(
            f"compiled model has {len(specs)} input(s); CYWS needs {len(INPUT_NAMES)} "
            f"({', '.join(INPUT_NAMES)})"
        )
    notes: list[str] = []
    for name, spec in zip(INPUT_NAMES, specs):
        shape = _spec_shape(spec)
        if len(shape) == 4 and shape[0] == 1:
            shape = shape[1:]
        if not shape:
            notes.append(f"{name}: model reports no input shape; assuming {IN_SHAPE}")
        elif shape == (3, MODEL_SIZE, MODEL_SIZE):
            raise CywsError(
                f"{name}: model expects CHW {shape}; the app feeds HWC {IN_SHAPE}. "
                "Recompile with NHWC inputs."
            )
        elif shape != IN_SHAPE:
            raise CywsError(f"{name}: model input shape {_spec_shape(spec)}, expected {IN_SHAPE}")
        dtypes = [_enum_name(d) for d in (getattr(spec, "dtypes", None) or [])]
        if dtypes and not any(d.lower() in ("float32", "fp32") for d in dtypes):
            raise CywsError(f"{name}: model input dtypes {dtypes}, expected Float32")
    return notes


def _reshape_by_spec(arr: np.ndarray, spec: Any) -> np.ndarray:
    """Give a flat runtime output its shape from the model's output spec.

    ``Model.output_specs()`` returns ``TensorSpec`` (= ``TensorConstraint``),
    whose ``shape`` is the static output shape (-1 marks a dynamic axis); the
    height / width / depth path is kept for runtimes that hand out an
    ``OutputSpec`` instead.
    """
    if arr.ndim >= 3 or spec is None:
        return arr
    try:
        shape = [int(v) for v in (getattr(spec, "shape", None) or [])]
    except (TypeError, ValueError):
        shape = []
    if len(shape) >= 3 and all(v > 0 for v in shape) and int(np.prod(shape)) == arr.size:
        return arr.reshape(shape)
    try:
        if hasattr(spec, "has_shape") and not spec.has_shape():
            return arr
        h, w, d = int(spec.height), int(spec.width), int(spec.depth)
    except (AttributeError, TypeError, ValueError):
        return arr
    if h <= 0 or w <= 0 or d <= 0 or h * w * d != arr.size:
        return arr
    if "CHW" in _enum_name(getattr(spec, "layout", "")).upper():
        return arr.reshape(d, h, w)
    return arr.reshape(h, w, d)


def _sample_tensors(sample: Any) -> list[Any]:
    """Flatten a pyneat Sample (tensor / tensors / bundle fields) or list into its tensors."""
    if sample is None:
        return []
    if isinstance(sample, (list, tuple)):
        return list(sample)
    tensors = list(getattr(sample, "tensors", None) or [])
    if not tensors and getattr(sample, "tensor", None) is not None:
        tensors = [sample.tensor]
    if not tensors:
        for f in getattr(sample, "fields", None) or []:
            tensors.extend(_sample_tensors(f))
    return tensors


def _measure_dict(report: Any) -> dict:
    """The parts of a pyneat MeasureReport worth logging."""
    e2e = getattr(report, "end_to_end", None)
    out: dict[str, Any] = {}
    if e2e is not None:
        out["end_to_end"] = {
            k: float(getattr(e2e, k)) for k in ("count", "avg_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "max_ms")
            if hasattr(e2e, k)
        }
    for k in ("throughput_inferences_per_s", "elapsed_s"):
        if hasattr(report, k):
            out[k] = float(getattr(report, k))
    warnings = getattr(report, "warnings", None)
    if warnings:
        out["warnings"] = [str(w) for w in warnings]
    return out


class NeatBackend(Backend):
    """Compiled MPK on the Modalix MLA via pyneat: Model -> Graph -> Run, else ModelRunner.

    Every build / push / pull / close holds ``common.ACCELERATOR_LOCK``: building
    or closing a runner while another one is inside run() fails the running one.
    """

    name = "neat"

    def __init__(
        self,
        model_path: str | Path,
        *,
        use_graph: bool = True,
        max_in_flight: int = 2,
        timeout_ms: int = 10000,
        measure: bool = False,
    ) -> None:
        super().__init__(max_in_flight)
        path = Path(model_path)
        if not path.is_file():
            raise CywsError(f"compiled model not found: {path}")
        if int(timeout_ms) <= 0:
            raise CywsError(f"timeout_ms must be > 0, got {timeout_ms}")
        try:
            import pyneat  # noqa: PLC0415 - board-only dependency
        except ImportError as exc:
            raise CywsError(f"pyneat is not importable ({exc}); the neat backend runs on the DevKit") from None
        self._pn = pyneat
        self.model_path = path
        self.timeout_ms = int(timeout_ms)
        self.mode = ""
        self.graph_error = ""
        self.spec_notes: list[str] = []
        self._exec: Any = None
        self._graph: Any = None
        self._model: Any = None
        self._tags: deque[Hashable] = deque()
        self._scope: Any = None
        self._measure_report: Optional[dict] = None
        self._measure_error = ""
        self._output_specs: list[Any] = []

        opt = pyneat.ModelOptions()
        try:
            opt.preprocess.kind = pyneat.InputKind.Tensor  # raw tensors in, no image preprocess
        except AttributeError:
            pass
        with common.ACCELERATOR_LOCK:
            try:
                self._model = pyneat.Model(str(path), opt)
            except Exception as exc:  # noqa: BLE001 - surface the runtime message
                raise CywsError(f"pyneat could not load {path}: {exc}") from exc
        try:
            self._build(use_graph)
        except BaseException:
            self._close()  # release the Model under the lock, not whenever GC gets to it
            raise
        if measure:
            self._start_measurement()

    @property
    def cache_tag(self) -> str:
        """Identifies the compiled model (and with it the precision), e.g. neat-cyws_bf16_mpk."""
        return model_cache_tag(self.name, self.model_path)

    # -- building
    def _build(self, use_graph: bool) -> None:
        self.spec_notes = _validate_input_specs(self._model.input_specs())
        try:
            self._output_specs = list(self._model.output_specs())
        except Exception:  # noqa: BLE001 - informational only
            self._output_specs = []
        seeds = [np.zeros(IN_SHAPE, np.float32), np.zeros(IN_SHAPE, np.float32)]
        path = self.model_path
        if use_graph:
            try:
                self._exec = self._build_graph(seeds)
                self.mode = "graph"
            except Exception as exc:  # noqa: BLE001 - any Graph failure -> ModelRunner
                self.graph_error = f"{type(exc).__name__}: {exc}"
                self._exec = None
        if self._exec is None:
            try:
                self._exec = self._build_runner(seeds)
                self.mode = "runner"
            except Exception as exc:  # noqa: BLE001
                hint = f" (graph path failed first: {self.graph_error})" if self.graph_error else ""
                raise CywsError(f"pyneat could not build a runner for {path}: {exc}{hint}") from exc

    def _tensor(self, hwc: np.ndarray) -> Any:
        t = self._pn.Tensor.from_numpy(np.ascontiguousarray(hwc, dtype=np.float32), copy=True)
        t.layout = self._pn.TensorLayout.HWC
        return t

    def _warm_up(self, exe: Any, seeds: list[np.ndarray], graph: bool) -> None:
        """One real round trip, so a pipeline that builds but cannot run falls back too."""
        self._push(exe, seeds[0], seeds[1])
        self._pull(exe, graph)

    def _build_graph(self, seeds: list[np.ndarray]) -> Any:
        pn = self._pn
        run = None
        try:
            with common.ACCELERATOR_LOCK:
                # Keep the Graph alive next to its Run, so it is only ever released under the lock.
                self._graph = pn.Graph()
                ro = pn.ModelRouteOptions()
                ro.include_input = True
                ro.include_output = True
                self._graph.add(self._model.graph(ro))
                run = self._graph.build([self._tensor(s) for s in seeds])
            self._warm_up(run, seeds, graph=True)
            return run
        except BaseException:
            if run is not None:
                self._shutdown(run)
            with common.ACCELERATOR_LOCK:
                self._graph = None
            raise

    def _build_runner(self, seeds: list[np.ndarray]) -> Any:
        with common.ACCELERATOR_LOCK:
            runner = self._model.build([self._tensor(s) for s in seeds])
        try:
            self._warm_up(runner, seeds, graph=False)
        except BaseException:
            self._shutdown(runner)
            raise
        return runner

    def _shutdown(self, exe: Any) -> None:
        with common.ACCELERATOR_LOCK:
            for method in ("close_input", "close"):
                fn = getattr(exe, method, None)
                if fn is None:
                    continue
                try:
                    fn()
                except Exception:  # noqa: BLE001 - best effort on teardown
                    pass

    # -- push / pull
    def _last_error(self, exe: Any) -> str:
        fn = getattr(exe, "last_error", None)
        try:
            return str(fn()) if fn else ""
        except Exception:  # noqa: BLE001
            return ""

    def _runtime_error(self, exe: Any, what: str, exc: Exception) -> CywsError:
        """pyneat raises plain RuntimeError (e.g. infra.accelerator_execution_failed);
        as a CywsError it fails only the current pair, and run_dataset keeps going."""
        err = self._last_error(exe)
        return CywsError(f"pyneat {what} failed: {type(exc).__name__}: {exc}"
                         + (f" (last error: {err})" if err and err not in str(exc) else ""))

    def _push(self, exe: Any, before: np.ndarray, after: np.ndarray) -> None:
        tensors = [self._tensor(before), self._tensor(after)]  # order = INPUT_NAMES
        with common.ACCELERATOR_LOCK:
            try:
                ok = exe.push_tensors(tensors)
            except Exception as exc:  # noqa: BLE001 - any runtime fault is a per-pair failure
                raise self._runtime_error(exe, "push", exc) from exc
            if ok is False:
                raise CywsError(f"pyneat push failed: {self._last_error(exe) or 'input closed'}")

    def _pull(self, exe: Any, graph: bool) -> np.ndarray:
        with common.ACCELERATOR_LOCK:
            try:
                if graph:
                    tensors = list(exe.pull_tensors(self.timeout_ms) or [])
                else:
                    tensors = _sample_tensors(exe.pull(self.timeout_ms))
                # Copy right away: to_numpy() views the pipeline's small output buffer
                # pool, and holding on to it stalls the pipeline after a few frames.
                arrays = [np.array(t.to_numpy(), np.float32, copy=True) for t in tensors]
                names = [str(getattr(getattr(t, "route", None), "name", "") or "") for t in tensors]
            except Exception as exc:  # noqa: BLE001 - any runtime fault is a per-pair failure
                raise self._runtime_error(exe, "pull", exc) from exc
            if not arrays:
                raise CywsError(
                    f"pyneat produced no output within {self.timeout_ms} ms"
                    + (f": {err}" if (err := self._last_error(exe)) else "")
                )
            del tensors
        if len(arrays) == len(self._output_specs):
            arrays = [_reshape_by_spec(a, s) for a, s in zip(arrays, self._output_specs)]
        arrays = _order_split_by_names(arrays, names)
        return to_canonical(arrays if len(arrays) > 1 else arrays[0])

    def _submit(self, tag: Hashable, before: np.ndarray, after: np.ndarray) -> None:
        if self._exec is None:
            raise CywsError("neat backend has no runner (closed, or released by benchmark())")
        self._push(self._exec, before, after)
        self._tags.append(tag)

    def _collect(self) -> tuple[Hashable, np.ndarray]:
        if not self._tags:
            raise CywsError("neat: collect without a pending push")
        out = self._pull(self._exec, self.mode == "graph")
        return self._tags.popleft(), out

    def _abort(self, exc: BaseException) -> None:
        # A failed push/pull leaves an unknown number of results in the pipeline,
        # so later outputs could be paired with the wrong window: refuse further use.
        self._broken = f"{type(exc).__name__}: {exc}"
        self._tags.clear()

    # -- measurement / benchmark
    def _start_measurement(self) -> None:
        try:
            with common.ACCELERATOR_LOCK:
                opts = self._pn.MeasureOptions() if hasattr(self._pn, "MeasureOptions") else None
                self._scope = (
                    self._exec.start_measurement(opts) if opts is not None
                    else self._exec.start_measurement(False)
                )
        except Exception as exc:  # noqa: BLE001 - measurement is optional
            self._measure_error = f"{type(exc).__name__}: {exc}"
            self._scope = None

    def _stop_measurement(self) -> None:
        if self._scope is None:
            return
        scope, self._scope = self._scope, None
        try:
            with common.ACCELERATOR_LOCK:
                self._measure_report = _measure_dict(scope.stop())
        except Exception as exc:  # noqa: BLE001
            self._measure_error = f"{type(exc).__name__}: {exc}"

    def stats(self) -> dict:
        """Base stats plus pyneat details; stops the measurement scope (call once, at the end)."""
        self._stop_measurement()
        s = super().stats()
        s.update({
            "mode": self.mode,
            "model_path": str(self.model_path),
            "timeout_ms": self.timeout_ms,
        })
        if self.graph_error:
            s["graph_error"] = self.graph_error
        if self.spec_notes:
            s["spec_notes"] = list(self.spec_notes)
        if self._measure_report is not None:
            s["measure"] = self._measure_report
        if self._measure_error:
            s["measure_error"] = self._measure_error
        return s

    def _release_runner(self) -> None:
        """Stop measuring and shut down the Run / ModelRunner, keeping the Model."""
        self._stop_measurement()
        exe, self._exec = self._exec, None
        self._tags.clear()
        if exe is not None:
            self._shutdown(exe)
        with common.ACCELERATOR_LOCK:
            self._graph = None

    def benchmark(self, num_samples: int = 200) -> dict:
        """NEAT's own model benchmark (synthetic inputs, plugin latency included).

        Model.benchmark builds its own pipelines for the same MPK, and two
        pipelines on the MLA at once can fail or wedge it, so the app's runner
        is shut down first (by dropping the last reference, ``del runner``).
        The backend cannot infer afterwards: call this last, before close().
        """
        self._check_usable()
        if self._streaming:
            raise CywsError("cannot benchmark while a stream is running")
        if int(num_samples) < 1:
            raise CywsError(f"num_samples must be >= 1, got {num_samples}")
        pn = self._pn
        self._release_runner()
        with common.ACCELERATOR_LOCK:
            opts = pn.BenchmarkOptions()
            opts.num_samples = int(num_samples)
            if hasattr(opts, "include_plugin_latency"):
                opts.include_plugin_latency = True
            report = self._model.benchmark(opts)
        out: dict[str, Any] = {"num_samples": int(num_samples)}
        for k in ("latency_ms", "fps", "avg_power_watts", "energy_joules"):
            if hasattr(report, k):
                out[k] = float(getattr(report, k))
        return out

    def _close(self) -> None:
        self._release_runner()
        with common.ACCELERATOR_LOCK:
            self._model = None


# --------------------------------------------------------------------------- host backends

class OnnxBackend(_SyncBackend):
    """FP32 reference on the host with onnxruntime."""

    name = "onnx"

    def __init__(self, onnx_path: str | Path, providers: Optional[Sequence[str]] = None) -> None:
        super().__init__(max_in_flight=1)
        path = Path(onnx_path)
        if not path.is_file():
            raise CywsError(f"ONNX model not found: {path}")
        try:
            import onnxruntime as ort  # noqa: PLC0415 - host-only dependency
        except ImportError as exc:
            raise CywsError(f"onnxruntime is not importable ({exc}); the onnx backend runs on the host") from None
        available = ort.get_available_providers()
        if providers is None:
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        try:
            self._sess = ort.InferenceSession(str(path), providers=list(providers))
        except Exception as exc:  # noqa: BLE001
            raise CywsError(f"onnxruntime could not load {path}: {exc}") from exc
        inputs = {i.name: i for i in self._sess.get_inputs()}
        missing = [n for n in INPUT_NAMES if n not in inputs]
        if missing:
            raise CywsError(f"{path.name}: inputs {sorted(inputs)} lack {missing}; expected {list(INPUT_NAMES)}")
        # NCHW is the export default; accept an NHWC export too (last dim 3).
        self._nhwc = {n: (list(inputs[n].shape) or [None])[-1] == 3 for n in INPUT_NAMES}
        self._out_names = [o.name for o in self._sess.get_outputs()]
        self.model_path = path

    @property
    def cache_tag(self) -> str:
        """Identifies the ONNX file, e.g. onnx-cyws_fused (the namespace eval/sweep.py uses)."""
        return model_cache_tag(self.name, self.model_path)

    def _feed(self, name: str, hwc: np.ndarray) -> np.ndarray:
        x = hwc[None] if self._nhwc[name] else hwc.transpose(2, 0, 1)[None]
        return np.ascontiguousarray(x, dtype=np.float32)

    def _run(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        feed = {INPUT_NAMES[0]: self._feed(INPUT_NAMES[0], before),
                INPUT_NAMES[1]: self._feed(INPUT_NAMES[1], after)}
        outs = [np.asarray(o) for o in self._sess.run(None, feed)]
        outs = _order_split_by_names(outs, self._out_names)
        return to_canonical(outs if len(outs) > 1 else outs[0])

    def stats(self) -> dict:
        s = super().stats()
        s["providers"] = list(self._sess.get_providers())
        return s

    def _close(self) -> None:
        self._sess = None


class AfeBackend(_SyncBackend):
    """Quantized model in the SiMa Model Compiler host simulator (SDK container only)."""

    name = "afe"

    def __init__(self, model_name: str, model_dir: str | Path) -> None:
        super().__init__(max_in_flight=1)
        mdir = Path(model_dir)
        if not mdir.is_dir():
            raise CywsError(f"afe model directory not found: {mdir}")
        if not (mdir / f"{model_name}.sima").is_file():
            raise CywsError(f"afe model {model_name}.sima not found in {mdir}")
        try:
            from afe.apis.defines import InputName  # noqa: PLC0415 - SDK-container-only
            from afe.apis.model import Model  # noqa: PLC0415
        except ImportError as exc:
            raise CywsError(
                f"afe is not importable ({exc}); the afe backend runs inside the Model Compiler container"
            ) from None
        self._input_name = InputName
        try:
            self._model = Model.load(model_name, str(mdir))
        except Exception as exc:  # noqa: BLE001
            raise CywsError(f"afe could not load {model_name} from {mdir}: {exc}") from exc
        self.model_name = model_name

    @property
    def cache_tag(self) -> str:
        """Identifies the quantized model, e.g. afe-cyws_bf16."""
        return f"{self.name}-{self.model_name}"

    def _run(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        feed = {self._input_name(INPUT_NAMES[0]): before[None], self._input_name(INPUT_NAMES[1]): after[None]}
        outs = [np.asarray(o) for o in self._model.execute(feed, fast_mode=True)]
        return to_canonical(outs if len(outs) > 1 else outs[0])

    def _close(self) -> None:
        self._model = None


class FakeBackend(_SyncBackend):
    """Deterministic stand-in for the model: heat peaks on blobs of large |before - after|.

    Lets the whole app (tiling, decode, merge, masks, CLI) run without a model.
    Each blob gets a Gaussian peak at its bbox centre in both branches, with wh
    equal to the bbox size and the sub-pixel offset of the centre, so a decode
    recovers the bbox.
    """

    name = "fake"

    def __init__(self, seed: int = 0, *, diff_thr: float = 0.1, min_area: int = 9, max_in_flight: int = 2) -> None:
        super().__init__(max_in_flight)
        if not 0.0 < diff_thr < 1.0:
            raise CywsError(f"diff_thr must be in (0, 1), got {diff_thr}")
        self.seed = int(seed)
        self.diff_thr = float(diff_thr)
        self.min_area = int(min_area)
        # A low, seeded noise floor (< 0.02) keeps the heat realistic but never above a threshold.
        self._floor = np.random.default_rng(self.seed).uniform(0.0, 0.02, (MODEL_SIZE, MODEL_SIZE)).astype(np.float32)
        self._yy, self._xx = np.mgrid[0:MODEL_SIZE, 0:MODEL_SIZE].astype(np.float32)

    def _run(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        diff = np.abs(after - before).mean(axis=2).astype(np.float32)
        diff = cv2.GaussianBlur(diff, (5, 5), 1.0)
        mask = (diff > self.diff_thr).astype(np.uint8)
        n, labels, boxes, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = np.zeros((N_OUT, MODEL_SIZE, MODEL_SIZE), np.float32)
        hb, (wbw, wbh), (obx, oby) = BRANCH_SLICES["before"]
        ha, (waw, wah), (oax, oay) = BRANCH_SLICES["after"]
        out[hb] = self._floor
        out[ha] = self._floor
        last = MODEL_SIZE - 1
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in boxes[i])
            if area < self.min_area:
                continue
            cx, cy = x + w / 2.0, y + h / 2.0
            ix, iy = min(int(cx), last), min(int(cy), last)
            strength = float(diff[labels == i].mean())
            peak = 0.5 + 0.49 * min(1.0, strength / 0.25)
            sigma = max(1.5, min(w, h) / 6.0)
            g = peak * np.exp(-((self._xx - ix) ** 2 + (self._yy - iy) ** 2) / (2.0 * sigma * sigma))
            np.maximum(out[ha], g, out=out[ha])
            np.maximum(out[hb], 0.95 * g, out=out[hb])
            r = slice(max(0, iy - 2), iy + 3), slice(max(0, ix - 2), ix + 3)
            for cw, ch_, ox, oy in ((waw, wah, oax, oay), (wbw, wbh, obx, oby)):
                out[cw][r] = w
                out[ch_][r] = h
                out[ox][r] = cx - ix
                out[oy][r] = cy - iy
        return out


# --------------------------------------------------------------------------- factory

def _resolve_afe(model_path: Optional[str | Path], model_name: Optional[str]) -> tuple[str, Path]:
    """`<dir>/<name>.sima` file, or a directory plus model_name, or a directory with one model."""
    if model_path is None:
        raise CywsError("the afe backend needs --model-path (a .sima file or its directory)")
    p = Path(model_path)
    if p.is_file():
        if p.suffix != ".sima":
            raise CywsError(f"afe model must be a .sima file, got {p}")
        return (model_name or p.name[: -len(".sima")]), p.parent
    if p.is_dir():
        if model_name:
            return model_name, p
        cands = sorted(c for c in p.glob("*.sima") if not c.name.endswith(".fp32.sima"))
        if len(cands) != 1:
            raise CywsError(f"{p}: found {len(cands)} quantized .sima models; pass model_name")
        return cands[0].name[: -len(".sima")], p
    raise CywsError(f"afe model path not found: {p}")


def make_backend(kind: str, model_path: Optional[str | Path] = None, **kw: Any) -> Backend:
    """Backend factory for the CLI: kind in neat | onnx | afe | fake."""
    kind = (kind or "").lower()
    if kind == "fake":
        return FakeBackend(**kw)
    if kind in ("neat", "onnx") and model_path is None:
        raise CywsError(f"the {kind} backend needs --model-path")
    if kind == "neat":
        return NeatBackend(model_path, **kw)
    if kind == "onnx":
        return OnnxBackend(model_path, **kw)
    if kind == "afe":
        name, mdir = _resolve_afe(model_path, kw.pop("model_name", None))
        return AfeBackend(name, mdir, **kw)
    raise CywsError(f"unknown backend {kind!r}; choose neat, onnx, afe or fake")
