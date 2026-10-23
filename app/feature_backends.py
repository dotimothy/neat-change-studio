"""DINOv2 feature extractors for the zero-shot change model (app/zeroshot.py).

Every backend takes one image as (S, S, 3) float32 RGB / 255 and returns (T, 384) float32,
T = 5 + G*G: the class token, 4 registers, then the patch tokens row by row (G = S / 14).
`split()` turns that into the class token and a (G, G, 384) patch map.

NeatFeatureBackend follows DINOv2/app/dinov2_backends.py's NeatBackend, but takes
common.ACCELERATOR_LOCK (shared with the CYWS backends) around build, push, pull and
close, so two pyneat runners never overlap. pyneat is imported only when it is built.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

import common
from backends import _percentiles
from common import CywsError

DIM, NREG, PATCH = 384, 4, 14


def split(feat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(T, 384) -> class token (384,), patch map (G, G, 384)."""
    f = np.asarray(feat, np.float32).reshape(-1, DIM)
    g = int(round((f.shape[0] - 1 - NREG) ** 0.5))
    if 1 + NREG + g * g != f.shape[0]:
        raise ValueError(f"{f.shape[0]} tokens is not 5 + G*G")
    return f[0], f[1 + NREG:].reshape(g, g, DIM)


class FeatureBackend:
    """Base: input check and latency stats like the CYWS backends' stats()."""

    name = "features"
    size = 448

    def __init__(self) -> None:
        self._latencies_ms: list[float] = []
        self._closed = False

    def _features(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def features(self, rgb01: np.ndarray) -> np.ndarray:
        if self._closed:
            raise CywsError(f"{self.name} backend is closed")
        x = np.ascontiguousarray(rgb01, np.float32)
        if x.shape != (self.size, self.size, 3):
            raise CywsError(f"expected ({self.size}, {self.size}, 3), got {x.shape}")
        t0 = time.perf_counter()
        y = self._features(x)
        self._latencies_ms.append((time.perf_counter() - t0) * 1e3)
        del self._latencies_ms[:-1000]
        return y

    def stats(self) -> dict:
        return {"backend": self.name, "inferences": len(self._latencies_ms),
                "latency": _percentiles(self._latencies_ms)}

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._close()

    def _close(self) -> None:
        pass


class NeatFeatureBackend(FeatureBackend):
    """The compiled MPK on the Modalix MLA, through pyneat (rank-3 HWC tensor in)."""

    name = "neat"

    def __init__(self, mpk: str | Path, timeout_ms: int = 10000) -> None:
        super().__init__()
        path = Path(mpk)
        if not path.is_file():
            raise CywsError(f"compiled model not found: {path}")
        try:
            import pyneat  # noqa: PLC0415 - board-only dependency
        except ImportError as exc:
            raise CywsError(f"pyneat is not importable ({exc}); the neat backend runs on the DevKit") from None
        self.pyneat, self.timeout_ms, self.run = pyneat, int(timeout_ms), None
        with common.ACCELERATOR_LOCK:
            opt = pyneat.ModelOptions()
            opt.preprocess.kind = pyneat.InputKind.Tensor       # raw tensors in, no image preprocessing
            self.model = pyneat.Model(str(path), opt)
            spec = self.model.input_specs()[0]
            dims = [d for d in getattr(spec, "shape", getattr(spec, "dims", [])) if d != 1] or [448, 448, 3]
            self.size = int(dims[0])
            self.run = self.model.build([self._tensor(np.zeros((self.size, self.size, 3), np.float32))])

    def _tensor(self, hwc: np.ndarray):
        t = self.pyneat.Tensor.from_numpy(np.ascontiguousarray(hwc, np.float32), copy=True)
        t.layout = self.pyneat.TensorLayout.HWC
        return t

    def _features(self, x: np.ndarray) -> np.ndarray:
        with common.ACCELERATOR_LOCK:
            self.run.push_tensors([self._tensor(x)])
            sample = self.run.pull(self.timeout_ms)
            if sample is None:
                raise CywsError(f"The MLA returned nothing within {self.timeout_ms / 1000:.0f} s")
            t = sample.tensors[0] if getattr(sample, "tensors", None) else sample.tensor
            y = np.array(t.to_numpy(), np.float32, copy=True)   # copy out: views hold pipeline buffers
        # NCHW (1, 384, 1, T) comes back as HWC (1, T, 384) or (T, 384); either way channels are last
        if y.shape[-1] != DIM and y.shape[0] == DIM:
            y = y.reshape(DIM, -1).T
        return y.reshape(-1, DIM)

    def _close(self) -> None:
        with common.ACCELERATOR_LOCK:
            if self.run is not None:
                try:
                    self.run.close()
                finally:
                    self.run = None
            self.model = None


class OnnxFeatureBackend(FeatureBackend):
    """The exported DINOv2 ONNX graph in ONNX Runtime FP32 (for a PC)."""

    name = "onnx"

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        if not Path(path).is_file():
            raise CywsError(f"DINOv2 ONNX model not found: {path}")
        import onnxruntime as ort  # noqa: PLC0415
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name, self.size = inp.name, int(inp.shape[2])

    def _features(self, x: np.ndarray) -> np.ndarray:
        y = self.sess.run(None, {self.input_name: x.transpose(2, 0, 1)[None]})[0]
        return np.asarray(y[0, :, 0].T, np.float32).copy()


class FakeFeatureBackend(FeatureBackend):
    """Deterministic stand-in for DINOv2 (--backend fake), no model needed.

    Each 14 x 14 patch is described by its colour mean and spread and its gradient
    strength, and that vector goes through fixed random Fourier features, so the cosine
    between two patches is about exp(-d^2 / 2 sigma^2): the same content gives the same
    features (no change), and a pasted object gives patches unlike the background.
    """

    name = "fake"

    def __init__(self, size: int = 448, seed: int = 0, sigma: float = 0.12) -> None:
        super().__init__()
        if size % PATCH:
            raise CywsError(f"size must be a multiple of {PATCH}, got {size}")
        self.size = int(size)
        rng = np.random.default_rng(seed)
        self._w = rng.normal(0.0, 1.0 / sigma, (7, DIM)).astype(np.float32)
        self._b = rng.uniform(0.0, 2 * np.pi, DIM).astype(np.float32)

    def _features(self, x: np.ndarray) -> np.ndarray:
        g = self.size // PATCH
        mean = cv2.resize(x, (g, g), interpolation=cv2.INTER_AREA)
        sq = cv2.resize(x * x, (g, g), interpolation=cv2.INTER_AREA)
        std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
        gray = x.mean(axis=2)
        grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        edge = cv2.resize(grad, (g, g), interpolation=cv2.INTER_AREA)[..., None] / 4.0
        v = np.concatenate([mean, std, edge], axis=-1).reshape(-1, 7)
        patches = np.cos(v @ self._w + self._b) * np.sqrt(2.0 / DIM)
        head = np.tile(patches.mean(axis=0, keepdims=True), (1 + NREG, 1))   # class + registers
        return np.concatenate([head, patches], axis=0).astype(np.float32)
