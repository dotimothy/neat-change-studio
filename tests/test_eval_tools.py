"""Tests of the host tuning tools in eval/: TorchBackend, CachedBackend and the sweep harness.

The torch tests need CUDA and the checkpoint / reference outputs in build/ and
are skipped otherwise; the cache and sweep tests run anywhere (FakeBackend).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

import backends  # noqa: E402
from cached_backend import CachedBackend, input_key  # noqa: E402
from common import CywsError  # noqa: E402

CKPT = ROOT / "build/ckpt/pretrained-resnet50-3x-coam-scSE-affine.ckpt"
EVAL_INPUTS = ROOT / "build/calib/eval_inputs.npz"
FP32_REF = ROOT / "build/eval/fp32.npy"


def _cuda() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


needs_cuda = pytest.mark.skipif(
    not (_cuda() and CKPT.is_file() and EVAL_INPUTS.is_file() and FP32_REF.is_file()),
    reason="needs torch with CUDA plus the checkpoint and build/eval/fp32.npy",
)


def _inputs(n: int, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        b = rng.random((256, 256, 3), dtype=np.float32)
        a = b.copy()
        y, x = rng.integers(20, 200, 2)
        a[y:y + 40, x:x + 40] = 1.0 - a[y:y + 40, x:x + 40]
        out.append((b, a))
    return out


class CountingBackend(backends.FakeBackend):
    def __init__(self) -> None:
        super().__init__(max_in_flight=3)
        self.calls = 0

    def _run(self, before, after):
        self.calls += 1
        return super()._run(before, after)


# ------------------------------------------------------------------ CachedBackend

def test_cache_miss_then_hit_identical(tmp_path):
    inner = CountingBackend()
    data = _inputs(5)
    with CachedBackend(inner, tmp_path, batch=4) as be:
        cold = [o for _, o in be.infer_stream((i, b, a) for i, (b, a) in enumerate(data))]
        assert inner.calls == 5 and be.misses == 5 and be.hits == 0
        warm = [o for _, o in be.infer_stream((i, b, a) for i, (b, a) in enumerate(data))]
        assert inner.calls == 5 and be.hits == 5
    for c, w in zip(cold, warm):
        assert c.dtype == np.float32 and c.shape == (10, 256, 256)
        np.testing.assert_array_equal(c, w)  # cold results are served float16-rounded too
    ref = backends.FakeBackend()
    np.testing.assert_allclose(cold[0], ref.infer(*data[0]), atol=2e-3, rtol=1e-3)
    assert len(list((tmp_path / "fake").rglob("*.npy"))) == 5


def test_cache_mixed_hits_keep_fifo_order(tmp_path):
    data = _inputs(7, seed=1)
    with CachedBackend(CountingBackend(), tmp_path, batch=3) as be:
        for i in (1, 4, 5):  # warm a few
            be.infer(*data[i])
        tags = [f"t{i}" for i in range(7)]
        got = list(be.infer_stream((t, b, a) for t, (b, a) in zip(tags, data)))
        assert [t for t, _ in got] == tags
        assert be.inner.calls == 7
        for (_, out), (b, a) in zip(got, data):
            np.testing.assert_array_equal(out, be.lookup(b, a))


def test_cache_key_is_exact_and_namespaced(tmp_path):
    (b, a), = _inputs(1)
    k = input_key(b, a, "x")
    b2 = b.copy()
    b2[0, 0, 0] = np.nextafter(b2[0, 0, 0], np.float32(2))
    assert input_key(b2, a, "x") != k
    assert input_key(a, b, "x") != k          # before/after order matters
    assert input_key(b, a, "y") != k
    with CachedBackend(CountingBackend(), tmp_path, namespace="ns1") as be:
        be.infer(b, a)
    with CachedBackend(CountingBackend(), tmp_path, namespace="ns2") as be:
        be.infer(b, a)
        assert be.misses == 1


def test_cache_only_raises_on_miss_and_recovers_corrupt(tmp_path):
    (b, a), = _inputs(1)
    with CachedBackend(None, tmp_path, namespace="fake") as be:
        with pytest.raises(CywsError, match="cache miss"):
            be.infer(b, a)
    with CachedBackend(CountingBackend(), tmp_path) as be:
        be.infer(b, a)
        p = be.path_for(input_key(b, a, "fake"))
    p.write_bytes(b"garbage")
    with CachedBackend(CountingBackend(), tmp_path) as be:
        be.infer(b, a)
        assert be.misses == 1 and be.inner.calls == 1


class _StubOnnx(backends.OnnxBackend):
    """OnnxBackend without onnxruntime: each 'model file' scales the fake heat differently."""

    def __init__(self, path: Path, gain: float) -> None:
        backends._SyncBackend.__init__(self, max_in_flight=1)
        self.model_path = Path(path)
        self._fake = backends.FakeBackend()
        self._gain = np.float32(gain)

    def _run(self, before, after):
        return self._fake._run(before, after) * self._gain

    def _close(self) -> None:
        pass


def test_model_cache_tag_names_the_model_file():
    assert backends.model_cache_tag("neat", "build/bf16/cyws_bf16_mpk.tar.gz") == "neat-cyws_bf16_mpk"
    assert backends.model_cache_tag("neat", "/x/cyws_w8a16_mpk.tar.gz") == "neat-cyws_w8a16_mpk"
    assert backends.model_cache_tag("onnx", "build/onnx/cyws_fused.onnx") == "onnx-cyws_fused"
    neat_a = object.__new__(backends.NeatBackend)
    neat_a.model_path = Path("cyws_bf16_mpk.tar.gz")
    neat_b = object.__new__(backends.NeatBackend)
    neat_b.model_path = Path("cyws_w8a16_mpk.tar.gz")
    assert neat_a.cache_tag != neat_b.cache_tag
    afe = object.__new__(backends.AfeBackend)
    afe.model_name = "cyws_bf16"
    assert afe.cache_tag == "afe-cyws_bf16"


def test_cache_default_namespace_separates_models(tmp_path):
    # Regression: the default namespace was inner.name ('onnx' / 'neat'), so a second model
    # file got the first model's cached outputs for identical inputs.
    (b, a), = _inputs(1)
    with CachedBackend(_StubOnnx(tmp_path / "m_fp32.onnx", 1.0), tmp_path / "c") as be:
        out_a = be.infer(b, a)
        assert be.namespace == "onnx-m_fp32" and be.misses == 1
    with CachedBackend(_StubOnnx(tmp_path / "m_other.onnx", 0.5), tmp_path / "c") as be:
        out_b = be.infer(b, a)
        assert be.namespace == "onnx-m_other" and be.misses == 1 and be.hits == 0
    assert not np.allclose(out_a, out_b)
    with CachedBackend(_StubOnnx(tmp_path / "m_fp32.onnx", 1.0), tmp_path / "c") as be:
        np.testing.assert_array_equal(be.infer(b, a), out_a)
        assert be.hits == 1


# ------------------------------------------------------------------ TorchBackend

@needs_cuda
def test_torch_matches_ort_fp32():
    from torch_backend import TorchBackend, verify_against_reference

    with TorchBackend(CKPT, batch=8) as be:
        res = verify_against_reference(be, EVAL_INPUTS, FP32_REF, limit=24)
        assert be.stats()["forward_calls"] == 3
    assert res["heat_max_abs"] < 1e-3, res


@needs_cuda
def test_torch_behind_cache(tmp_path):
    from torch_backend import TorchBackend

    z = np.load(EVAL_INPUTS)
    hwc = lambda x: np.ascontiguousarray(x.transpose(1, 2, 0), np.float32)  # noqa: E731
    items = [(i, hwc(z["in0_before"][i]), hwc(z["in1_after"][i])) for i in range(5)]
    ref = np.load(FP32_REF, mmap_mode="r")
    with CachedBackend(TorchBackend(CKPT, batch=4), tmp_path, batch=4) as be:
        assert be.namespace.startswith("torch-fp32-")
        cold = dict(be.infer_stream(items))
        warm = dict(be.infer_stream(items))
        assert be.hits == 5 and be.misses == 5
    for i in range(5):
        np.testing.assert_array_equal(cold[i], warm[i])
        assert np.abs(warm[i][:2] - ref[i][:2].astype(np.float32)).max() < 1e-3


# ------------------------------------------------------------------ sweep harness

def _write_dataset(root: Path) -> None:
    from test_app import gt_mask, make_scene, paste_change

    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    for k, change in enumerate((True, False)):
        before = make_scene(seed=k)
        after = paste_change(before) if change else before.copy()
        Image.fromarray(before).save(root / "images" / f"p{k}_ref.png")
        Image.fromarray(after).save(root / "images" / f"p{k}_cap.png")
        m = gt_mask() if change else np.zeros_like(gt_mask())
        Image.fromarray((m * 255).astype(np.uint8)).save(root / "masks" / f"p{k}.png")


@pytest.mark.parametrize("workers", [0, 2])
def test_sweep_end_to_end_fake(tmp_path, workers):
    import csv

    import sweep

    data = tmp_path / "data"
    _write_dataset(data)
    out = tmp_path / "out"
    rc = sweep.main(["--backend", "fake", "--dataset", str(data), "--out", str(out), "--cache", str(tmp_path / "c"),
                     "--levels", "0;0,1", "--tile", "auto,0.5", "--tiles", "full,gated", "--thr", "0.2,0.3,0.5",
                     "--reg", "off,on-diff", "--coarse-thr", "0.3", "--refine-top", "1",
                     "--workers", str(workers)])
    assert rc == 0
    rows = list(csv.DictReader((out / "sweep.csv").open()))
    # grid: L0 x 2 regs + L01 x 2 tiles x 2 tiles-modes x 2 regs = 10 structures; coarse = 10, refine = 2
    assert len(rows) == 12
    assert {r["structure"] for r in rows if r["thr"] != "0.3"} == {rows[0]["structure"]}
    best = rows[0]
    assert float(best["f1"]) > 0.5 and best["empty_pairs"] == "1" and best["failed"] == "0"
    md = (out / "sweep.md").read_text()
    assert "Per-pair breakdown" in md and "Skipped by pruning (18)" in md
    pairs = list(csv.DictReader((out / "sweep_pairs.csv").open()))
    assert len(pairs) == 12 * 2
    assert not list(out.glob(".pairs-*"))  # worker scratch removed
