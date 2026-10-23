"""C-3PO change segmentation backends for app/c3po_app.py.

Every backend takes the before and the after image, each (S, S, 3) float32 RGB / 255
(the ImageNet mean/std is inside the model), and returns the logits at a quarter of the
input, (C, S/4, S/4) float32, class 0 = no change. `change_logit()` turns them into one
map whose sigmoid is the change probability 1 - softmax(logits)[0]; it is linear in the
logits for two classes, so it can be upsampled before the sigmoid, as the original model
upsamples its logits.

NeatC3POBackend runs the compiled MPK (github.com/DoctorKey/C-3PO, compiled in
MLSandbox/C3PO) through pyneat, with common.ACCELERATOR_LOCK around build, push, pull and
close like the other neat backends. pyneat is imported only when it is built.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

import common
from backends import _percentiles
from common import CywsError

SIZE, CLASSES = 256, 2


def change_logit(logits: np.ndarray) -> np.ndarray:
    """(C, h, w) logits -> (h, w) map with sigmoid(map) = 1 - softmax(logits)[0]."""
    lg = np.asarray(logits, np.float32)
    if lg.shape[0] == 2:
        return lg[1] - lg[0]
    rest = lg[1:]
    m = rest.max(axis=0)
    return (m + np.log(np.exp(rest - m).sum(axis=0)) - lg[0]).astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))).astype(np.float32)


def to_chw(y: np.ndarray, size: int, classes: int = CLASSES) -> np.ndarray:
    """The model's output in whatever layout the runtime hands back -> (C, S/4, S/4).
    NCHW (1, C, g, g) usually comes back from pyneat as HWC (g, g, C). Size-1 axes are
    dropped first, so (1, C, g*g) is planar and (1, g*g, C) channels last; a bare flat
    vector is taken as channels last (pyneat's HWC). Any other shape is an error."""
    g = size // 4
    y = np.asarray(y, np.float32)
    if y.size != classes * g * g:
        raise CywsError(f"C-3PO output has {y.size} values, expected {classes} x {g} x {g}")
    shape = y.shape
    y = y.reshape([d for d in y.shape if d != 1] or [y.size])
    if y.shape == (classes, g, g):
        return np.ascontiguousarray(y)
    if y.shape == (g, g, classes):
        return np.ascontiguousarray(y.transpose(2, 0, 1))
    if y.shape == (classes, g * g):                                  # planar, flattened spatially
        return np.ascontiguousarray(y.reshape(classes, g, g))
    if y.shape == (g * g, classes) or y.ndim == 1:                   # channels last, flattened
        return np.ascontiguousarray(y.reshape(g, g, classes).transpose(2, 0, 1))
    raise CywsError(f"C-3PO output shape {shape} is not a layout of ({classes}, {g}, {g})")


def input_size(spec) -> int:
    """The square input side from a pyneat input spec: its `shape` (attribute or method) or
    `dims`, in any layout (HWC, CHW, with a batch axis); SIZE when the spec says nothing."""
    dims = getattr(spec, "shape", None)
    if dims is None:
        dims = getattr(spec, "dims", None)
    if callable(dims):
        dims = dims()
    dims = [int(d) for d in (dims if dims is not None else []) if int(d) not in (1, 3)]
    if not dims:
        return SIZE
    size = max(dims)
    if size % 4 or any(d != size for d in dims):
        raise CywsError(f"C-3PO input spec {list(dims)} is not a square (S, S, 3) image with S a multiple of 4")
    return size


class C3POBackend:
    """Base: input checks and latency stats like the other backends' stats()."""

    name = "c3po"
    size = SIZE
    classes = CLASSES

    def __init__(self) -> None:
        self._latencies_ms: list[float] = []
        self._closed = False

    def _infer(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _infer_many(self, pairs: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
        return [self._infer(b, a) for b, a in pairs]

    def _check(self, role: str, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, np.float32)
        if x.shape != (self.size, self.size, 3):
            raise CywsError(f"{role}: expected ({self.size}, {self.size}, 3), got {x.shape}")
        return x

    def infer_many(self, pairs) -> list[np.ndarray]:
        """Logits (C, S/4, S/4) for each (before, after) pair, in order. A runtime that can
        queue (the neat backend) keeps more than one pair in flight. Each pair's recorded
        latency is the batch's wall time over its size: the real per-pair cost."""
        if self._closed:
            raise CywsError(f"{self.name} backend is closed")
        xs = [(self._check("before", b), self._check("after", a)) for b, a in pairs]
        if not xs:
            return []
        t0 = time.perf_counter()
        ys = self._infer_many(xs)
        per = (time.perf_counter() - t0) * 1e3 / len(xs)
        self._latencies_ms.extend([per] * len(xs))
        del self._latencies_ms[:-1000]
        return ys

    def infer(self, before_rgb01: np.ndarray, after_rgb01: np.ndarray) -> np.ndarray:
        """Logits (C, S/4, S/4) for one before/after pair."""
        return self.infer_many([(before_rgb01, after_rgb01)])[0]

    def probability(self, before_rgb01: np.ndarray, after_rgb01: np.ndarray) -> np.ndarray:
        """Change probability (S/4, S/4)."""
        return sigmoid(change_logit(self.infer(before_rgb01, after_rgb01)))

    def stats(self) -> dict:
        return {"backend": self.name, "inferences": len(self._latencies_ms),
                "latency": _percentiles(self._latencies_ms)}

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._close()

    def _close(self) -> None:
        pass


class NeatC3POBackend(C3POBackend):
    """The compiled MPK on the Modalix MLA, through pyneat (two rank-3 HWC tensors in, before then after)."""

    name = "neat"

    def __init__(self, mpk: str | Path, timeout_ms: int = 10000, max_in_flight: int = 2) -> None:
        super().__init__()
        path = Path(mpk)
        if not path.is_file():
            raise CywsError(f"compiled model not found: {path}")
        try:
            import pyneat  # noqa: PLC0415 - board-only dependency
        except ImportError as exc:
            raise CywsError(f"pyneat is not importable ({exc}); the neat backend runs on the DevKit") from None
        self.pyneat, self.timeout_ms, self.run, self.model = pyneat, int(timeout_ms), None, None
        self.max_in_flight = max(1, int(max_in_flight))
        with common.ACCELERATOR_LOCK:
            opt = pyneat.ModelOptions()
            opt.preprocess.kind = pyneat.InputKind.Tensor       # raw tensors in, no image preprocessing
            self.model = pyneat.Model(str(path), opt)
            specs = list(self.model.input_specs())
            if len(specs) != 2:
                self.model = None
                raise CywsError(f"{path.name} has {len(specs)} inputs; a C-3PO model takes two (before, after)")
            self.size = input_size(specs[0])
            z = np.zeros((self.size, self.size, 3), np.float32)
            self.run = self.model.build([self._tensor(z), self._tensor(z)])

    def _tensor(self, hwc: np.ndarray):
        t = self.pyneat.Tensor.from_numpy(np.ascontiguousarray(hwc, np.float32), copy=True)
        t.layout = self.pyneat.TensorLayout.HWC
        return t

    def _pull(self) -> np.ndarray:
        sample = self.run.pull(self.timeout_ms)
        if sample is None:
            raise CywsError(f"The MLA returned nothing within {self.timeout_ms / 1000:.0f} s")
        t = sample.tensors[0] if getattr(sample, "tensors", None) else sample.tensor
        return np.array(t.to_numpy(), np.float32, copy=True)   # copy out: views hold pipeline buffers

    def _infer_many(self, pairs: list[tuple[np.ndarray, np.ndarray]]) -> list[np.ndarray]:
        """Pushes run ahead of pulls by up to max_in_flight pairs (like the CYWS NeatBackend),
        so the host-side work on one pair overlaps the MLA on the previous one."""
        outs: list[np.ndarray] = []
        with common.ACCELERATOR_LOCK:
            pending = 0
            try:
                for before, after in pairs:
                    if pending >= self.max_in_flight:
                        outs.append(self._pull())
                        pending -= 1
                    self.run.push_tensors([self._tensor(before), self._tensor(after)])
                    pending += 1
                while pending:
                    outs.append(self._pull())
                    pending -= 1
            finally:
                for _ in range(pending):          # an error part way: drain so the next call starts in sync
                    try:
                        self.run.pull(self.timeout_ms)
                    except Exception:  # noqa: BLE001
                        break
        return [to_chw(y, self.size, self.classes) for y in outs]

    def _infer(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        return self._infer_many([(before, after)])[0]

    def _close(self) -> None:
        with common.ACCELERATOR_LOCK:
            if self.run is not None:
                try:
                    self.run.close()
                finally:
                    self.run = None
            self.model = None


class OnnxC3POBackend(C3POBackend):
    """The exported C-3PO ONNX graph (MLSandbox/C3PO build/onnx) in ONNX Runtime FP32 (for a PC)."""

    name = "onnx"

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        if not Path(path).is_file():
            raise CywsError(f"C-3PO ONNX model not found: {path}")
        import onnxruntime as ort  # noqa: PLC0415
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = self.sess.get_inputs()
        if len(inputs) != 2:
            raise CywsError(f"{Path(path).name} has {len(inputs)} inputs; a C-3PO model takes two (before, after)")
        self.names = [i.name for i in inputs]
        self.size = int(inputs[0].shape[2])
        out = self.sess.get_outputs()[0].shape
        self.classes = int(out[1]) if isinstance(out[1], int) else CLASSES

    def _infer(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        feed = {n: x.transpose(2, 0, 1)[None] for n, x in zip(self.names, (before, after))}
        y = self.sess.run(None, feed)[0]
        return np.asarray(y[0], np.float32).copy()


class FakeC3POBackend(C3POBackend):
    """Deterministic stand-in for C-3PO (--backend fake), no model needed: the change logit
    grows with the colour difference of the two images, averaged over 4 x 4 pixels."""

    name = "fake"

    def __init__(self, size: int = SIZE, gain: float = 40.0, offset: float = 0.08) -> None:
        super().__init__()
        if size % 4:
            raise CywsError(f"size must be a multiple of 4, got {size}")
        self.size, self.gain, self.offset = int(size), float(gain), float(offset)

    def _infer(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        g = self.size // 4
        d = np.abs(cv2.resize(after, (g, g), interpolation=cv2.INTER_AREA)
                   - cv2.resize(before, (g, g), interpolation=cv2.INTER_AREA)).mean(axis=2)
        d = cv2.GaussianBlur(d, (0, 0), 0.7)
        change = self.gain * (d - self.offset)
        return np.stack([np.zeros_like(change), change]).astype(np.float32)
