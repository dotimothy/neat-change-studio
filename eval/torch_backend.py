"""Host GPU backend: the PyTorch CYWS model (model/cyws_mla.py) on CUDA, for fast sweeps.

FP32 by default (TF32 disabled, so it tracks the ONNX FP32 reference); ``bf16=True``
runs under ``torch.autocast(bfloat16)`` as a rough stand-in for the MLA's bf16.
Implements the app's Backend contract (backends.Backend): FIFO ``infer_stream``
of (tag, before_hwc, after_hwc) -> canonical float32 (10, 256, 256).

Batching: ``_submit`` only queues; a forward pass runs when the base class
collects (``max_in_flight`` pairs queued, or the stream ends), so a stream of N
windows costs ceil(N / batch) GPU calls.

Host-only (torch); never deployed to the board.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Hashable, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "app", ROOT / "model"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import backends  # noqa: E402
from common import MODEL_SIZE, OUT_CHANNELS, CywsError  # noqa: E402

DEFAULT_CKPT = ROOT / "build/ckpt/pretrained-resnet50-3x-coam-scSE-affine.ckpt"


def cache_tag_for(ckpt: str | Path, bf16: bool) -> str:
    """Cache namespace of a TorchBackend (no torch import needed), so FP32 and bf16 never mix."""
    return f"torch-{'bf16' if bf16 else 'fp32'}-{Path(ckpt).stem}"


class TorchBackend(backends.Backend):
    """cyws_mla.build(ckpt) (BN folded, head fused: the exported graph) on a torch device."""

    name = "torch"

    def __init__(
        self,
        ckpt: str | Path = DEFAULT_CKPT,
        *,
        device: str = "cuda",
        bf16: bool = False,
        batch: int = 16,
        channels_last: bool = True,
    ) -> None:
        super().__init__(max_in_flight=max(1, int(batch)))
        try:
            import torch  # noqa: PLC0415 - host-only dependency
        except ImportError as exc:
            raise CywsError(f"torch is not importable ({exc}); the torch backend runs on the host") from None
        import cyws_mla  # noqa: PLC0415 - imports torch

        ckpt = Path(ckpt)
        if not ckpt.is_file():
            raise CywsError(f"checkpoint not found: {ckpt}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise CywsError("CUDA is not available; pass device='cpu' (slow) or use the onnx backend")
        # FP32 means FP32: TF32 convs/matmuls drift ~1e-3 from the ONNX reference.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # No cudnn autotuning: the GPU server sees every batch size 1..batch, and tuning each
        # new shape costs seconds; steady-state speed is within ~10% without it.
        torch.backends.cudnn.benchmark = False
        self._torch = torch
        self.device = torch.device(device)
        self.bf16 = bool(bf16)
        self.batch = int(batch)
        self.ckpt = ckpt
        model = cyws_mla.build(str(ckpt), export=True, outputs="fused")
        model = model.to(self.device).eval()
        self._mem_fmt = torch.channels_last if channels_last and self.device.type == "cuda" else torch.contiguous_format
        model = model.to(memory_format=self._mem_fmt)
        self._model = model
        self._queue: deque[tuple[Hashable, np.ndarray, np.ndarray]] = deque()
        self._ready: deque[tuple[Hashable, np.ndarray]] = deque()
        self._forward_calls = 0
        if self.bf16:
            self.name = "torch-bf16"

    @property
    def cache_tag(self) -> str:
        """Identifies the numerics, so a cache never mixes FP32 and bf16 outputs."""
        return cache_tag_for(self.ckpt, self.bf16)

    # -- batch forward ------------------------------------------------------------
    def run_batch(self, befores: Sequence[np.ndarray], afters: Sequence[np.ndarray]) -> np.ndarray:
        """(B, 256, 256, 3) x 2 HWC float32 -> (B, 10, 256, 256) float32 canonical."""
        torch = self._torch
        xb = torch.from_numpy(np.stack(befores).astype(np.float32, copy=False))
        xa = torch.from_numpy(np.stack(afters).astype(np.float32, copy=False))
        # NHWC numpy -> NCHW logical view; channels_last keeps NHWC storage on the GPU.
        xb = xb.to(self.device, non_blocking=True).permute(0, 3, 1, 2).contiguous(memory_format=self._mem_fmt)
        xa = xa.to(self.device, non_blocking=True).permute(0, 3, 1, 2).contiguous(memory_format=self._mem_fmt)
        with torch.inference_mode():
            if self.bf16:
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    out = self._model(xb, xa)
            else:
                out = self._model(xb, xa)
        self._forward_calls += 1
        out = out.float().contiguous().cpu().numpy()
        if out.shape[1:] != (len(OUT_CHANNELS), MODEL_SIZE, MODEL_SIZE):
            raise CywsError(f"torch model output {out.shape}, expected (B, 10, 256, 256)")
        return out

    # -- Backend hooks ------------------------------------------------------------
    def _submit(self, tag: Hashable, before: np.ndarray, after: np.ndarray) -> None:
        self._queue.append((tag, before, after))

    def _flush(self) -> None:
        while self._queue:
            chunk = [self._queue.popleft() for _ in range(min(self.batch, len(self._queue)))]
            outs = self.run_batch([c[1] for c in chunk], [c[2] for c in chunk])
            for (tag, _, _), out in zip(chunk, outs):
                self._ready.append((tag, np.array(out, np.float32, copy=True)))

    def _collect(self) -> tuple[Hashable, np.ndarray]:
        if not self._ready:
            self._flush()
        if not self._ready:
            raise CywsError(f"{self.name}: collect without a pending submit")
        return self._ready.popleft()

    def _abort(self, exc: BaseException) -> None:
        self._queue.clear()
        self._ready.clear()

    def stats(self) -> dict:
        s = super().stats()
        s.update(device=str(self.device), bf16=self.bf16, batch=self.batch, forward_calls=self._forward_calls)
        return s

    def _close(self) -> None:
        self._model = None
        if self.device.type == "cuda":
            self._torch.cuda.empty_cache()


def verify_against_reference(
    be: backends.Backend,
    inputs: str | Path = ROOT / "build/calib/eval_inputs.npz",
    reference: str | Path = ROOT / "build/eval/fp32.npy",
    limit: Optional[int] = None,
) -> dict:
    """Run the fixed eval tensors through ``be`` and compare with the ORT FP32 outputs."""
    z = np.load(inputs)
    xb, xa = z["in0_before"], z["in1_after"]
    ref = np.load(reference, mmap_mode="r")
    n = len(xb) if limit is None else min(int(limit), len(xb))
    hwc = lambda x: np.ascontiguousarray(x.transpose(1, 2, 0), np.float32)  # noqa: E731
    heat_err = 0.0
    all_err = 0.0
    for i, out in be.infer_stream((i, hwc(xb[i]), hwc(xa[i])) for i in range(n)):
        r = np.asarray(ref[i], np.float32)
        heat_err = max(heat_err, float(np.abs(out[:2] - r[:2]).max()))
        all_err = max(all_err, float(np.abs(out - r).max()))
    return {"n": n, "heat_max_abs": heat_err, "all_max_abs": all_err}


if __name__ == "__main__":
    import argparse
    import json
    import time

    ap = argparse.ArgumentParser(description="Check TorchBackend against build/eval/fp32.npy (ORT FP32).")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    with TorchBackend(a.ckpt, bf16=a.bf16, batch=a.batch) as be:
        t0 = time.perf_counter()
        res = verify_against_reference(be, limit=a.limit)
        res["seconds"] = round(time.perf_counter() - t0, 2)
        res["stats"] = be.stats()
    print(json.dumps(res, indent=1))
