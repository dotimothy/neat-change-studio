"""Tests for app/backends.py: output layouts, streaming order, ONNX, fake and (mocked) pyneat."""

from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest

import backends
import common
from backends import (
    Backend,
    FakeBackend,
    NeatBackend,
    OnnxBackend,
    _resolve_afe,
    make_backend,
    to_canonical,
)
from common import CH, CywsError

S = common.MODEL_SIZE


def _canon(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((10, S, S), dtype=np.float32)


def _pair(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return (rng.random((S, S, 3), dtype=np.float32), rng.random((S, S, 3), dtype=np.float32))


# --------------------------------------------------------------------------- to_canonical

def _layouts(ref: np.ndarray) -> dict[str, object]:
    hwc = ref.transpose(1, 2, 0)
    parts = [ref[0:2], ref[2:6], ref[6:10]]
    return {
        "nchw": ref[None].copy(),
        "chw": ref.copy(),
        "hwc": np.ascontiguousarray(hwc),
        "nhwc": np.ascontiguousarray(hwc)[None],
        "split_chw": [p.copy() for p in parts],
        "split_nchw": [p[None].copy() for p in parts],
        "split_hwc": [np.ascontiguousarray(p.transpose(1, 2, 0)) for p in parts],
        "split_nhwc": tuple(np.ascontiguousarray(p.transpose(1, 2, 0))[None] for p in parts),
    }


@pytest.mark.parametrize("layout", list(_layouts(np.zeros((10, S, S), np.float32))))
def test_to_canonical_every_layout(layout):
    ref = _canon(1)
    src = _layouts(ref)[layout]
    out = to_canonical(src)
    assert out.shape == (10, S, S) and out.dtype == np.float32 and out.flags.c_contiguous
    np.testing.assert_array_equal(out, ref)
    for a in src if isinstance(src, (list, tuple)) else [src]:
        assert not np.shares_memory(out, a), "must never return a view of a runtime buffer"


def test_to_canonical_converts_dtype_and_copies_canonical_input():
    ref = _canon(2)
    out = to_canonical(ref.astype(np.float16))
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, ref, atol=1e-3)
    out2 = to_canonical(ref)
    assert not np.shares_memory(out2, ref)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((10, 128, 128), np.float32),
        np.zeros((2, 10, S, S), np.float32),
        np.zeros((S, S), np.float32),
        [np.zeros((2, S, S)), np.zeros((4, S, S))],
        [np.zeros((4, S, S)), np.zeros((4, S, S)), np.zeros((2, S, S))],
    ],
)
def test_to_canonical_rejects_bad_shapes(bad):
    with pytest.raises(CywsError):
        to_canonical(bad)


def test_order_split_by_names():
    h, w, o = np.zeros(1), np.ones(1), np.full(1, 2.0)
    got = backends._order_split_by_names([o, h, w], ["off", "heatmap", "wh"])
    assert [g[0] for g in got] == [0.0, 1.0, 2.0]
    # ambiguous names -> unchanged
    same = backends._order_split_by_names([o, h, w], ["out0", "out1", "out2"])
    assert [g[0] for g in same] == [2.0, 0.0, 1.0]


# --------------------------------------------------------------------------- streaming (FakeBackend)

class CountingFake(FakeBackend):
    """FakeBackend that records how many submits are outstanding."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.outstanding = 0
        self.peak = 0
        self.order: list = []

    def _submit(self, tag, before, after):
        super()._submit(tag, before, after)
        self.outstanding += 1
        self.peak = max(self.peak, self.outstanding)
        self.order.append(("submit", tag))

    def _collect(self):
        res = super()._collect()
        self.outstanding -= 1
        self.order.append(("collect", res[0]))
        return res


@pytest.mark.parametrize("mif", [1, 2, 3])
def test_infer_stream_fifo_and_max_in_flight(mif):
    be = CountingFake(max_in_flight=mif)
    pairs = [_pair(i) for i in range(7)]
    consumed = []

    def items():
        for i, (b, a) in enumerate(pairs):
            consumed.append(i)
            yield (f"t{i}", b, a)

    results = []
    for tag, out in be.infer_stream(items()):
        if not results:
            # The first result is only pulled once the window is full and one more item is read.
            assert len(consumed) == mif + 1
        results.append((tag, out))
    assert [t for t, _ in results] == [f"t{i}" for i in range(7)]
    assert be.peak == mif
    assert be.outstanding == 0
    ref = FakeBackend()
    for (_, out), (b, a) in zip(results, pairs):
        np.testing.assert_array_equal(out, ref.infer(b, a))
    st = be.stats()
    assert st["inferences"] == 7 and st["max_seen_in_flight"] == mif
    assert set(st["latency"]) >= {"mean_ms", "p50_ms", "p90_ms", "max_ms"}


def test_infer_stream_early_break_drains():
    be = CountingFake(max_in_flight=3)
    items = [(i, *_pair(i)) for i in range(6)]
    gen = be.infer_stream(items)
    assert next(gen)[0] == 0
    gen.close()
    assert be.outstanding == 0 and not be._results
    assert [t for t, _ in be.infer_stream(items[:2])] == [0, 1]


def test_infer_stream_bad_input_raises_and_recovers():
    be = CountingFake(max_in_flight=2)
    b, a = _pair(0)
    items = [(0, b, a), (1, b, a[:, :, :2])]
    with pytest.raises(CywsError, match="after input must have shape"):
        list(be.infer_stream(items))
    assert be.outstanding == 0
    assert [t for t, _ in be.infer_stream([(5, b, a)])] == [5]
    with pytest.raises(CywsError, match="float"):
        be.infer((b * 255).astype(np.uint8), a)
    with pytest.raises(CywsError, match=r"\(tag, before_hwc, after_hwc\)"):
        list(be.infer_stream([(1, b)]))


def test_tag_mismatch_is_asserted():
    class Swapping(FakeBackend):
        def _collect(self):
            tag, out = super()._collect()
            return ("wrong", out)

    be = Swapping()
    b, a = _pair(0)
    with pytest.raises(CywsError, match="expecting"):
        list(be.infer_stream([("right", b, a)]))
    with pytest.raises(CywsError, match="unusable"):
        be.infer(b, a)


def test_one_stream_at_a_time_and_close():
    be = FakeBackend()
    b, a = _pair(0)
    g1 = be.infer_stream([(0, b, a), (1, b, a)])
    next(g1)
    with pytest.raises(CywsError, match="already streaming"):
        next(be.infer_stream([(2, b, a)]))
    g1.close()
    with be:
        be.infer(b, a)
    with pytest.raises(CywsError, match="closed"):
        be.infer(b, a)
    be.close()  # idempotent
    with pytest.raises(CywsError):
        FakeBackend(max_in_flight=0)


# --------------------------------------------------------------------------- FakeBackend output

def test_fake_backend_heat_peak_over_pasted_difference():
    rng = np.random.default_rng(3)
    before = np.clip(0.4 + 0.03 * rng.standard_normal((S, S, 3)), 0, 1).astype(np.float32)
    after = before.copy()
    x0, y0, w, h = 100, 60, 40, 30
    after[y0:y0 + h, x0:x0 + w] = (0.95, 0.1, 0.1)
    out = FakeBackend(seed=0).infer(before, after)
    assert out.shape == (10, S, S) and out.dtype == np.float32
    for heat_ch, wh, off in (common.BRANCH_SLICES["after"], common.BRANCH_SLICES["before"]):
        heat = out[heat_ch]
        py, px = np.unravel_index(int(np.argmax(heat)), heat.shape)
        cx, cy = px + out[off[0], py, px], py + out[off[1], py, px]
        assert abs(cx - (x0 + w / 2)) <= 2 and abs(cy - (y0 + h / 2)) <= 2
        assert heat[py, px] > 0.5
        assert abs(out[wh[0], py, px] - w) <= 6 and abs(out[wh[1], py, px] - h) <= 6
        far = heat.copy()
        far[y0 - 20:y0 + h + 20, x0 - 20:x0 + w + 20] = 0
        assert far.max() < 0.02
    np.testing.assert_array_equal(out, FakeBackend(seed=0).infer(before, after))
    assert not np.array_equal(out[CH["heat_a"]], FakeBackend(seed=1).infer(before, after)[CH["heat_a"]])
    flat = FakeBackend().infer(before, before)
    assert flat[CH["heat_a"]].max() < 0.02 and flat[CH["heat_b"]].max() < 0.02


# --------------------------------------------------------------------------- OnnxBackend

def _write_onnx(path: Path, split: bool) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    shape = [1, 3, S, S]
    ins = [helper.make_tensor_value_info(n, TensorProto.FLOAT, shape) for n in common.INPUT_NAMES]
    b, a = common.INPUT_NAMES
    nodes = [
        helper.make_node("Mul", [b, a], ["prod"]),
        helper.make_node("Sub", [b, a], ["diff"]),
        helper.make_node("ReduceMean", ["diff"], ["dmean"], axes=[1], keepdims=1),
        helper.make_node("ReduceMean", [b], ["bmean"], axes=[1], keepdims=1),
        helper.make_node("ReduceMean", [a], ["amean"], axes=[1], keepdims=1),
    ]
    if not split:
        nodes.append(helper.make_node("Concat", [b, a, "prod", "dmean"], ["out"], axis=1))
        outs = [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 10, S, S])]
    else:
        # NHWC split outputs, deliberately listed in the order off, heat, wh.
        nodes += [
            helper.make_node("Concat", ["bmean", "amean"], ["heat_c"], axis=1),
            helper.make_node("Concat", [b, "amean"], ["wh_c"], axis=1),
            helper.make_node("Concat", [a, "bmean"], ["off_c"], axis=1),
        ]
        outs = []
        for n, c in (("off", 4), ("heat", 2), ("wh", 4)):
            nodes.append(helper.make_node("Transpose", [f"{n}_c"], [n], perm=[0, 2, 3, 1]))
            outs.append(helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, S, S, c]))
    graph = helper.make_graph(nodes, "tiny_cyws", ins, outs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def test_onnx_backend_fused(tmp_path):
    pytest.importorskip("onnxruntime")
    p = tmp_path / "tiny.onnx"
    _write_onnx(p, split=False)
    b, a = _pair(4)
    with OnnxBackend(p, providers=["CPUExecutionProvider"]) as be:
        out = be.infer(b, a)
        bc, ac = b.transpose(2, 0, 1), a.transpose(2, 0, 1)
        ref = np.concatenate([bc, ac, bc * ac, (bc - ac).mean(0, keepdims=True)])
        np.testing.assert_allclose(out, ref, atol=1e-6)
        tags = [t for t, _ in be.infer_stream([(i, *_pair(i)) for i in range(3)])]
        assert tags == [0, 1, 2]
        assert be.stats()["providers"] == ["CPUExecutionProvider"]


def test_onnx_backend_split_reordered_by_name(tmp_path):
    pytest.importorskip("onnxruntime")
    p = tmp_path / "split.onnx"
    _write_onnx(p, split=True)
    b, a = _pair(5)
    out = OnnxBackend(p, providers=["CPUExecutionProvider"]).infer(b, a)
    bc, ac = b.transpose(2, 0, 1), a.transpose(2, 0, 1)
    bm, am = bc.mean(0, keepdims=True), ac.mean(0, keepdims=True)
    ref = np.concatenate([bm, am, bc, am, ac, bm])
    np.testing.assert_allclose(out, ref, atol=1e-6)


def test_onnx_backend_validation(tmp_path):
    pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    with pytest.raises(CywsError, match="not found"):
        OnnxBackend(tmp_path / "missing.onnx")
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, S, S])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, S, S])
    g = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "g", [x], [y])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    p = tmp_path / "wrong.onnx"
    onnx.save(m, str(p))
    with pytest.raises(CywsError, match="in0_before"):
        OnnxBackend(p, providers=["CPUExecutionProvider"])


# --------------------------------------------------------------------------- mocked pyneat

def _expected_canonical(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """What the fake MPK computes: channel c = mean(before) * (c + 1) + mean(after) * 100."""
    c = np.arange(10, dtype=np.float32)[:, None, None]
    return np.broadcast_to(before.mean() * (c + 1) + after.mean() * 100.0, (10, S, S)).astype(np.float32)


class FakePyneat:
    """A stand-in for the pyneat module that records calls and checks the accelerator lock."""

    def __init__(self, *, specs=None, graph_build_error=None, graph_pull_empty=False,
                 push_result=True, flat_outputs=False, output_specs=None):
        self.calls: list[str] = []
        self.violations: list[str] = []
        self.specs = specs if specs is not None else [self.spec((S, S, 3)), self.spec((S, S, 3))]
        self.graph_build_error = graph_build_error
        self.graph_pull_empty = graph_pull_empty
        self.push_result = push_result
        self.flat_outputs = flat_outputs
        self.output_specs = output_specs or []
        self.runs: list = []
        self.module = self._make_module()

    def spec(self, shape, dtypes=("Float32",)):
        return types.SimpleNamespace(shape=list(shape), dtypes=[types.SimpleNamespace(name=d) for d in dtypes])

    def record(self, what: str) -> None:
        self.calls.append(what)
        if not common.ACCELERATOR_LOCK._is_owned():
            self.violations.append(what)

    def _make_module(self) -> types.ModuleType:
        fp = self
        mod = types.ModuleType("pyneat")
        mod.InputKind = types.SimpleNamespace(Tensor="InputKind.Tensor", Image="InputKind.Image")
        mod.TensorLayout = types.SimpleNamespace(HWC="TensorLayout.HWC", CHW="TensorLayout.CHW")

        class ModelOptions:
            def __init__(self):
                self.preprocess = types.SimpleNamespace(kind=None)

        class ModelRouteOptions:
            include_input = False
            include_output = False

        class Tensor:
            def __init__(self, arr):
                self._arr = arr
                self.layout = None
                self.route = types.SimpleNamespace(name="")

            @classmethod
            def from_numpy(cls, arr, copy=False, layout=None):
                assert copy is True
                return cls(np.array(arr, copy=True))

            def to_numpy(self, copy=False):
                return self._arr

        class Pool:
            """One output buffer reused for every pull, like NEAT's small output pool."""

            def __init__(self):
                self.buf = np.zeros((S * S * 10,) if fp.flat_outputs else (S, S, 10), np.float32)

            def emit(self, before, after):
                hwc = _expected_canonical(before, after).transpose(1, 2, 0)
                self.buf[...] = hwc.reshape(self.buf.shape)
                return Tensor(self.buf)  # a view: the backend must copy it

        class _Exec:
            kind = "exec"

            def __init__(self, seeds):
                fp.runs.append(self)
                self.queue: list = []
                self.pool = Pool()
                self.closed: list[str] = []
                for t in seeds:
                    assert t.layout == mod.TensorLayout.HWC and t._arr.shape == (S, S, 3)

            def push_tensors(self, tensors):
                fp.record(f"{self.kind}.push")
                assert len(tensors) == 2 and all(t.layout == mod.TensorLayout.HWC for t in tensors)
                if fp.push_result:
                    self.queue.append((tensors[0]._arr, tensors[1]._arr))
                return fp.push_result

            def _next(self):
                b, a = self.queue.pop(0)
                return self.pool.emit(b, a)

            def start_measurement(self, opts=None):
                fp.record(f"{self.kind}.start_measurement")
                scope = types.SimpleNamespace()

                def stop():
                    fp.record("scope.stop")
                    e2e = types.SimpleNamespace(count=3, avg_ms=12.5, p50_ms=12.0, p90_ms=14.0,
                                                p95_ms=14.5, p99_ms=15.0, max_ms=16.0)
                    return types.SimpleNamespace(end_to_end=e2e, throughput_inferences_per_s=80.0,
                                                 elapsed_s=1.0, warnings=[])

                scope.stop = stop
                return scope

            def last_error(self):
                return "fake push rejected"

            def close_input(self):
                fp.record(f"{self.kind}.close_input")
                self.closed.append("close_input")

            def close(self):
                fp.record(f"{self.kind}.close")
                self.closed.append("close")

        class Run(_Exec):
            kind = "run"

            def pull_tensors(self, timeout_ms=-1):
                fp.record("run.pull_tensors")
                if fp.graph_pull_empty or not self.queue:
                    return []
                return [self._next()]

        class ModelRunner(_Exec):
            kind = "runner"

            def pull(self, timeout_ms=-1):
                fp.record("runner.pull")
                if not self.queue:
                    return types.SimpleNamespace(tensors=[], tensor=None, fields=[])
                return types.SimpleNamespace(tensors=[self._next()], tensor=None, fields=[])

        class Graph:
            def __init__(self):
                fp.record("Graph()")
                self.parts: list = []

            def add(self, part):
                fp.record("graph.add")
                self.parts.append(part)

            def build(self, inputs, options=None, copy=False, layout=None, image_format=None):
                fp.record("graph.build")
                if fp.graph_build_error:
                    raise RuntimeError(fp.graph_build_error)
                assert self.parts and self.parts[0].include_input and self.parts[0].include_output
                return Run(inputs)

        class BenchmarkOptions:
            num_samples = 100
            include_plugin_latency = False

        class Model:
            def __init__(self, path, opt):
                fp.record("Model()")
                fp.model_opt = opt
                fp.model_path = path

            def input_specs(self):
                return fp.specs

            def output_specs(self):
                return fp.output_specs

            def graph(self, ro):
                fp.record("model.graph")
                return ro

            def build(self, inputs, route_options=None, run_options=None, copy=False):
                fp.record("model.build")
                return ModelRunner(inputs)

            def benchmark(self, opts):
                fp.record("model.benchmark")
                fp.bench_opts = opts
                return types.SimpleNamespace(latency_ms=9.5, fps=105.0, avg_power_watts=4.0, energy_joules=0.1)

        mod.ModelOptions = ModelOptions
        mod.ModelRouteOptions = ModelRouteOptions
        mod.Tensor = Tensor
        mod.Graph = Graph
        mod.Model = Model
        mod.BenchmarkOptions = BenchmarkOptions
        mod.MeasureOptions = lambda: types.SimpleNamespace()
        return mod


@pytest.fixture
def mpk(tmp_path):
    p = tmp_path / "cyws_mpk.tar.gz"
    p.write_bytes(b"fake")
    return p


def _install(monkeypatch, fp: FakePyneat) -> FakePyneat:
    monkeypatch.setitem(sys.modules, "pyneat", fp.module)
    return fp


def _stream(be: Backend, n: int):
    pairs = [_pair(10 + i) for i in range(n)]
    results = list(be.infer_stream([(i, b, a) for i, (b, a) in enumerate(pairs)]))
    return pairs, results


def test_neat_graph_path(monkeypatch, mpk):
    fp = _install(monkeypatch, FakePyneat())
    be = NeatBackend(mpk, max_in_flight=2, timeout_ms=500)
    assert be.mode == "graph" and be.graph_error == ""
    assert fp.model_opt.preprocess.kind == fp.module.InputKind.Tensor
    assert fp.model_path == str(mpk)
    pairs, results = _stream(be, 5)
    assert [t for t, _ in results] == list(range(5))
    for (_, out), (b, a) in zip(results, pairs):
        # Outputs stay correct after later pulls overwrote the shared pool buffer: they were copied.
        np.testing.assert_allclose(out, _expected_canonical(b, a), rtol=1e-6)
        assert not np.shares_memory(out, fp.runs[0].pool.buf)
    assert "runner.pull" not in fp.calls and "model.build" not in fp.calls
    run = fp.runs[0]
    be.close()
    assert run.closed == ["close_input", "close"]
    assert fp.violations == [], f"called without ACCELERATOR_LOCK: {fp.violations}"
    with pytest.raises(CywsError, match="closed"):
        be.infer(*pairs[0])


def test_neat_falls_back_when_graph_build_raises(monkeypatch, mpk):
    fp = _install(monkeypatch, FakePyneat(graph_build_error="graph unsupported"))
    with NeatBackend(mpk, max_in_flight=3) as be:
        assert be.mode == "runner"
        assert "graph unsupported" in be.graph_error
        pairs, results = _stream(be, 4)
        for (_, out), (b, a) in zip(results, pairs):
            np.testing.assert_allclose(out, _expected_canonical(b, a), rtol=1e-6)
        assert be.stats()["graph_error"].startswith("RuntimeError")
    assert fp.runs[-1].closed == ["close_input", "close"]
    assert fp.violations == []


def test_neat_falls_back_when_graph_warmup_fails(monkeypatch, mpk):
    fp = _install(monkeypatch, FakePyneat(graph_pull_empty=True))
    be = NeatBackend(mpk, timeout_ms=50)
    assert be.mode == "runner" and "no output" in be.graph_error
    assert fp.runs[0].kind == "run" and fp.runs[0].closed == ["close_input", "close"]
    b, a = _pair(0)
    np.testing.assert_allclose(be.infer(b, a), _expected_canonical(b, a), rtol=1e-6)
    be.close()
    assert fp.violations == []


def test_neat_use_graph_false_and_flat_outputs(monkeypatch, mpk):
    spec = types.SimpleNamespace(height=S, width=S, depth=10, layout="TensorLayout.HWC", has_shape=lambda: True)
    fp = _install(monkeypatch, FakePyneat(flat_outputs=True, output_specs=[spec]))
    be = NeatBackend(mpk, use_graph=False)
    assert be.mode == "runner" and "Graph()" not in fp.calls
    b, a = _pair(1)
    np.testing.assert_allclose(be.infer(b, a), _expected_canonical(b, a), rtol=1e-6)
    be.close()


@pytest.mark.parametrize(
    "specs, match",
    [
        ("one", "1 input"),
        ("chw", "CHW"),
        ("small", "expected"),
        ("uint8", "Float32"),
    ],
)
def test_neat_input_spec_validation(monkeypatch, mpk, specs, match):
    fp = FakePyneat()
    table = {
        "one": [fp.spec((S, S, 3))],
        "chw": [fp.spec((3, S, S)), fp.spec((3, S, S))],
        "small": [fp.spec((224, 224, 3)), fp.spec((224, 224, 3))],
        "uint8": [fp.spec((S, S, 3), ("UInt8",)), fp.spec((S, S, 3), ("UInt8",))],
    }
    fp.specs = table[specs]
    _install(monkeypatch, fp)
    with pytest.raises(CywsError, match=match):
        NeatBackend(mpk)
    assert "graph.build" not in fp.calls and "model.build" not in fp.calls


def test_neat_accepts_batched_and_unknown_specs(monkeypatch, mpk):
    fp = FakePyneat()
    fp.specs = [fp.spec((1, S, S, 3)), fp.spec((), ())]
    _install(monkeypatch, fp)
    be = NeatBackend(mpk)
    assert be.stats()["spec_notes"] and "in1_after" in be.stats()["spec_notes"][0]
    be.close()


def test_neat_measure_and_benchmark(monkeypatch, mpk):
    fp = _install(monkeypatch, FakePyneat())
    be = NeatBackend(mpk, measure=True)
    _stream(be, 3)
    st = be.stats()
    assert st["measure"]["end_to_end"]["avg_ms"] == 12.5
    assert st["measure"]["throughput_inferences_per_s"] == 80.0
    assert st["inferences"] == 3 and st["mode"] == "graph"
    bench = be.benchmark(num_samples=50)
    assert bench == {"num_samples": 50, "latency_ms": 9.5, "fps": 105.0, "avg_power_watts": 4.0, "energy_joules": 0.1}
    assert fp.bench_opts.num_samples == 50 and fp.bench_opts.include_plugin_latency is True
    be.close()
    assert "run.start_measurement" in fp.calls and "scope.stop" in fp.calls
    assert fp.violations == []


def test_neat_push_failure_marks_backend_unusable(monkeypatch, mpk):
    fp = _install(monkeypatch, FakePyneat())
    be = NeatBackend(mpk)
    fp.push_result = False
    b, a = _pair(0)
    with pytest.raises(CywsError, match="fake push rejected"):
        be.infer(b, a)
    fp.push_result = True
    with pytest.raises(CywsError, match="unusable"):
        be.infer(b, a)
    be.close()


def test_neat_lock_blocks_other_threads_during_pull(monkeypatch, mpk):
    """While the backend pulls, another thread cannot take the accelerator lock."""
    fp = _install(monkeypatch, FakePyneat())
    be = NeatBackend(mpk)
    seen: list[bool] = []
    run_cls = type(fp.runs[0])
    orig = run_cls.pull_tensors

    def probing_pull(self, timeout_ms=-1):
        got = []
        t = threading.Thread(target=lambda: got.append(common.ACCELERATOR_LOCK.acquire(blocking=False)))
        t.start()
        t.join()
        seen.append(got[0])
        return orig(self, timeout_ms)

    monkeypatch.setattr(run_cls, "pull_tensors", probing_pull)
    be.infer(*_pair(0))
    assert seen == [False]
    be.close()


def test_neat_errors_without_pyneat_or_model(monkeypatch, tmp_path, mpk):
    with pytest.raises(CywsError, match="not found"):
        NeatBackend(tmp_path / "missing.tar.gz")
    monkeypatch.setitem(sys.modules, "pyneat", None)
    with pytest.raises(CywsError, match="pyneat is not importable"):
        NeatBackend(mpk)


# --------------------------------------------------------------------------- factory

def test_make_backend(tmp_path):
    assert isinstance(make_backend("fake", seed=3), FakeBackend)
    with pytest.raises(CywsError, match="unknown backend"):
        make_backend("tpu")
    with pytest.raises(CywsError, match="model-path"):
        make_backend("neat")
    with pytest.raises(CywsError, match="model-path"):
        make_backend("onnx")
    d = tmp_path / "sima"
    d.mkdir()
    (d / "cyws_bf16.sima").write_bytes(b"x")
    (d / "cyws_bf16.fp32.sima").write_bytes(b"x")
    assert _resolve_afe(d, None) == ("cyws_bf16", d)
    assert _resolve_afe(d / "cyws_bf16.sima", None) == ("cyws_bf16", d)
    assert _resolve_afe(d, "other") == ("other", d)
    with pytest.raises(CywsError, match="afe"):
        make_backend("afe", d)  # afe exists only in the Model Compiler container


@pytest.mark.parametrize("where", ["push", "pull", "to_numpy"])
def test_neat_runtime_errors_become_cyws_errors(monkeypatch, mpk, where):
    """pyneat raises plain RuntimeError on MLA faults; run_dataset only survives CywsError."""
    fp = _install(monkeypatch, FakePyneat())
    be = NeatBackend(mpk)
    run_cls = type(fp.runs[0])
    msg = "infra.accelerator_execution_failed"

    def boom(*_a, **_k):
        raise RuntimeError(msg)

    if where == "push":
        monkeypatch.setattr(run_cls, "push_tensors", boom)
    elif where == "pull":
        monkeypatch.setattr(run_cls, "pull_tensors", boom)
    else:
        monkeypatch.setattr(fp.module.Tensor, "to_numpy", boom)
    b, a = _pair(0)
    with pytest.raises(CywsError, match=msg) as info:
        be.infer(b, a)
    assert isinstance(info.value.__cause__, RuntimeError)
    with pytest.raises(CywsError, match="unusable"):
        be.infer(b, a)
    be.close()
    assert fp.violations == []


@pytest.mark.parametrize("shape", [(S, S, 10), (1, S, S, 10), (10, S, S)])
def test_neat_flat_output_reshaped_by_tensor_constraint(monkeypatch, mpk, shape):
    """Model.output_specs() returns TensorConstraint (dtypes / rank / shape), not OutputSpec."""
    spec = types.SimpleNamespace(dtypes=[], rank=len(shape), shape=list(shape))
    fp = _install(monkeypatch, FakePyneat(flat_outputs=True, output_specs=[spec]))
    be = NeatBackend(mpk, use_graph=False)
    if shape == (10, S, S):
        # The fake MPK writes HWC; make this runner write CHW instead.
        def emit_chw(pool, before, after):
            pool.buf[...] = _expected_canonical(before, after).reshape(-1)
            return fp.module.Tensor(pool.buf)

        pool = fp.runs[-1].pool
        monkeypatch.setattr(type(pool), "emit", emit_chw)
    b, a = _pair(2)
    np.testing.assert_allclose(be.infer(b, a), _expected_canonical(b, a), rtol=1e-6)
    be.close()


def test_neat_benchmark_shuts_down_runner_first(monkeypatch, mpk):
    """Model.benchmark builds its own pipelines; the app's Run must be gone by then."""
    fp = _install(monkeypatch, FakePyneat())
    be = NeatBackend(mpk, measure=True)
    _stream(be, 2)
    run = fp.runs[0]
    be.benchmark(num_samples=10)
    assert run.closed == ["close_input", "close"]
    assert fp.calls.index("run.close") < fp.calls.index("model.benchmark")
    assert fp.calls.index("scope.stop") < fp.calls.index("run.close")
    assert be.stats()["measure"]["end_to_end"]["avg_ms"] == 12.5  # report kept after release
    with pytest.raises(CywsError, match="no runner"):
        be.infer(*_pair(0))
    be.close()
    assert fp.calls.count("run.close") == 1
    assert fp.violations == []
