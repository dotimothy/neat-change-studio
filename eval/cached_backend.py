"""Disk cache in front of any app backend, keyed by the exact input bytes.

A sweep that only changes post-processing (thresholds, merge, masks, verify)
feeds the backend the very same crops again; this returns the stored output
instead of recomputing it. Key = blake2b(namespace, before.tobytes(), after.tobytes())
over the float32 (256, 256, 3) tensors, so any change to cropping, resizing,
registration or decode resolution is a miss, never a stale hit.

Outputs are stored as float16 .npy (1.3 MB each) under
``cache_dir/<namespace>/<key[:2]>/<key>.npy``; float16 keeps heat within 5e-4 of
FP32. The namespace defaults to the inner backend's ``cache_tag``, which names the
model file or checkpoint and precision (``onnx-cyws_fused``, ``neat-cyws_bf16_mpk``,
torch ckpt stem + fp32/bf16), so outputs of different models never mix. Only a
backend without a ``cache_tag`` falls back to its name; give such a backend an
explicit namespace whenever it can load more than one model. Overwriting a
model file in place keeps its tag: clear or rename the namespace then. Writes are atomic (tmp + rename), so several
processes can share one cache directory.

Misses are sent to the inner backend in batches (up to ``batch`` per call to
inner.infer_stream), results come back in input order.
"""

from __future__ import annotations

import hashlib
import os
import sys
from collections import deque
from pathlib import Path
from typing import Hashable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "app") not in sys.path:
    sys.path.insert(0, str(ROOT / "app"))

import backends  # noqa: E402
from common import MODEL_SIZE, OUT_CHANNELS, CywsError  # noqa: E402

OUT_SHAPE = (len(OUT_CHANNELS), MODEL_SIZE, MODEL_SIZE)


def input_key(before: np.ndarray, after: np.ndarray, namespace: str = "") -> str:
    """Hash of the exact float32 input bytes (plus the namespace)."""
    h = hashlib.blake2b(digest_size=20)
    h.update(namespace.encode())
    for x in (before, after):
        x = np.ascontiguousarray(x, dtype=np.float32)
        h.update(str(x.shape).encode())
        h.update(memoryview(x).cast("B"))
    return h.hexdigest()


class CachedBackend(backends.Backend):
    """Wraps ``inner`` (owned: closed with this backend unless own_inner=False).

    inner=None makes a cache-only backend: a miss raises CywsError.
    """

    def __init__(
        self,
        inner: Optional[backends.Backend],
        cache_dir: str | Path,
        *,
        namespace: Optional[str] = None,
        batch: int = 16,
        own_inner: bool = True,
        read_only: bool = False,
    ) -> None:
        super().__init__(max_in_flight=max(1, int(batch)))
        if namespace is None:
            if inner is None:
                raise CywsError("a cache-only CachedBackend needs an explicit namespace")
            namespace = str(getattr(inner, "cache_tag", None) or inner.name)
        self.inner = inner
        self.namespace = namespace
        self.dir = Path(cache_dir) / namespace
        self.dir.mkdir(parents=True, exist_ok=True)
        self.own_inner = own_inner
        self.read_only = read_only
        self.name = f"cached({inner.name if inner is not None else namespace})"
        # FIFO of [tag, key, out-or-None, before, after]; misses keep their inputs until computed.
        self._pending: deque[list] = deque()
        self.hits = 0
        self.misses = 0
        self.write_errors = 0

    def path_for(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.npy"

    def lookup(self, before: np.ndarray, after: np.ndarray) -> Optional[np.ndarray]:
        """Cached canonical output for these inputs, or None."""
        return self._load(self.path_for(input_key(before, after, self.namespace)))

    def _load(self, path: Path) -> Optional[np.ndarray]:
        try:
            arr = np.load(path)
        except FileNotFoundError:
            return None
        except (OSError, ValueError, EOFError):  # truncated / corrupt entry: recompute
            return None
        if arr.shape != OUT_SHAPE:
            return None
        return arr.astype(np.float32)

    def _store(self, key: str, out: np.ndarray) -> None:
        if self.read_only:
            return
        path = self.path_for(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            with open(tmp, "wb") as f:
                np.save(f, np.asarray(out, dtype=np.float16))
            os.replace(tmp, path)
        except OSError:
            self.write_errors += 1

    # -- Backend hooks ------------------------------------------------------------
    def _submit(self, tag: Hashable, before: np.ndarray, after: np.ndarray) -> None:
        key = input_key(before, after, self.namespace)
        out = self._load(self.path_for(key))
        if out is not None:
            self.hits += 1
            self._pending.append([tag, key, out, None, None])
        else:
            self.misses += 1
            self._pending.append([tag, key, None, before, after])

    def _compute_misses(self) -> None:
        todo = [e for e in self._pending if e[2] is None]
        if not todo:
            return
        if self.inner is None:
            raise CywsError(f"{self.name}: cache miss for key {todo[0][1]} and no inner backend")
        items = ((i, e[3], e[4]) for i, e in enumerate(todo))
        for i, out in self.inner.infer_stream(items):
            e = todo[i]
            # Serve exactly what a later hit would return, so cold and warm runs agree bit for bit.
            out16 = np.asarray(out, dtype=np.float16)
            self._store(e[1], out16)
            e[2], e[3], e[4] = out16.astype(np.float32), None, None

    def _collect(self) -> tuple[Hashable, np.ndarray]:
        if not self._pending:
            raise CywsError(f"{self.name}: collect without a pending submit")
        if self._pending[0][2] is None:
            self._compute_misses()
        tag, _, out, _, _ = self._pending.popleft()
        return tag, out

    def _abort(self, exc: BaseException) -> None:
        self._pending.clear()

    def stats(self) -> dict:
        s = super().stats()
        s.update(cache_dir=str(self.dir), hits=self.hits, misses=self.misses, write_errors=self.write_errors)
        if self.inner is not None:
            s["inner"] = self.inner.stats()
        return s

    def _close(self) -> None:
        if self.inner is not None and self.own_inner:
            self.inner.close()
