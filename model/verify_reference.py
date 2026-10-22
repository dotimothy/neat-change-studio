#!/usr/bin/env python3
"""Check model/cyws_mla.py against upstream CYWS, numerically.

Reference = upstream code, unmodified: the forked segmentation_models_pytorch
Unet (commit 0092ee4), upstream models/coattention.py, and a verbatim copy of
mmdet 2.x CenterNetHead.forward_single + decode (model/ref_centernet.py). Both
models load the same checkpoint with strict=True.

Checks (pass criteria from the plan):
  1. unfolded rewrite vs reference   float64: max|d| <= 1e-9 (exact up to rounding);
                                     float32: heat <= 1e-5, wh/off <= 1e-4 relative
  2. folded+fused vs unfolded        heat <= 1e-4, wh/off <= 1e-3 relative, same strong peaks,
                                     decoded boxes IoU > 0.999
  3. synthetic paste pairs           pasted patch detected in the after branch, score >= 0.3, IoU >= 0.5
Also logs co-attention logit ranges and runs a torch.autocast(bfloat16) pass on
the folded model as an early read of bf16 accuracy (boxes vs fp32).

Usage: verify_reference.py [--ckpt ...] [--data DIR]  (default: $CYWS_DATA or data/testCYWS) [--device cuda]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import cyws_mla  # noqa: E402
from ref_centernet import RefCenterNetHead, decode_heatmap  # noqa: E402

CKPT = ROOT / "build/ckpt/pretrained-resnet50-3x-coam-scSE-affine.ckpt"
SMP = ROOT / "third_party/smp/segmentation_models_pytorch"
CYWS = ROOT / "third_party/cyws"


# --------------------------------------------------------------------------- upstream loader
def _load(name: str, path: Path, package_dir: Path | None = None):
    """Load a module from a file; with package_dir it is a real package (relative imports work)."""
    locs = [str(package_dir)] if package_dir else None
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=locs)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _package(name: str, path: Path) -> types.ModuleType:
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    sys.modules[name] = pkg
    return pkg


def load_upstream_unet_cls():
    """Import the fork's Unet without executing smp/encoders/__init__ (it needs timm,
    pretrainedmodels, efficientnet_pytorch). Only the resnet encoder is wired up."""
    pm = types.ModuleType("pretrainedmodels")
    pmm = types.ModuleType("pretrainedmodels.models")
    pmt = types.ModuleType("pretrainedmodels.models.torchvision_models")
    pmt.pretrained_settings = {k: {"imagenet": {}} for k in
                               ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")}
    sys.modules.update({"pretrainedmodels": pm, "pretrainedmodels.models": pmm,
                        "pretrainedmodels.models.torchvision_models": pmt})
    _package("segmentation_models_pytorch", SMP)
    _load("segmentation_models_pytorch.base", SMP / "base" / "__init__.py", SMP / "base")
    enc = _package("segmentation_models_pytorch.encoders", SMP / "encoders")
    _load("segmentation_models_pytorch.encoders._utils", SMP / "encoders" / "_utils.py")
    _load("segmentation_models_pytorch.encoders._base", SMP / "encoders" / "_base.py")
    resnet = _load("segmentation_models_pytorch.encoders.resnet", SMP / "encoders" / "resnet.py")

    def get_encoder(name, in_channels=3, depth=5, weights=None, output_stride=32):
        assert weights is None, "reference is built without downloading imagenet weights"
        e = resnet.resnet_encoders[name]
        encoder = e["encoder"](**e["params"], depth=depth)
        encoder.set_in_channels(in_channels)
        return encoder

    enc.get_encoder = get_encoder
    _package("segmentation_models_pytorch.unet", SMP / "unet")
    return _load("segmentation_models_pytorch.unet.model", SMP / "unet" / "model.py").Unet


class RefModel(nn.Module):
    """Upstream CenterNetWithCoAttention.forward, minus Lightning/detectron2/mmdet."""

    def __init__(self):
        super().__init__()
        Unet = load_upstream_unet_cls()
        coam = _load("cyws_coattention", CYWS / "models" / "coattention.py")
        self.unet_model = Unet("resnet50", decoder_channels=(256, 256, 128, 128, 64), encoder_depth=5,
                               encoder_weights=None, num_coam_layers=3, decoder_attention_type="scse",
                               disable_segmentation_head=True)
        self.coattention_modules = nn.ModuleList(
            coam.CoAttentionModule(input_channels=c, hidden_channels=h, attention_type="coam")
            for c, h in ((2048, 256), (1024, 128), (512, 64)))
        self.centernet_head = RefCenterNetHead(64, 64, 1)

    def forward(self, left, right):
        lf = self.unet_model.encoder(left)
        rf = self.unet_model.encoder(right)
        for i in range(len(self.coattention_modules)):
            lf[-(i + 1)], rf[-(i + 1)] = self.coattention_modules[i](lf[-(i + 1)], rf[-(i + 1)])
        ld = self.unet_model.decoder(*lf)
        rd = self.unet_model.decoder(*rf)
        return self.centernet_head(ld), self.centernet_head(rd)


def ref_to_fused(outs) -> torch.Tensor:
    (hb, wb, ob), (ha, wa, oa) = outs
    return torch.cat([hb, ha, wb, wa, ob, oa], 1)


# --------------------------------------------------------------------------- inputs
def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1), np.float32) / 255.0)[None]


def squash(img: np.ndarray, s: int = 256) -> np.ndarray:
    return cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)


def load_rgb(p: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def dataset_inputs(root: Path, n_tiles: int, rng: np.random.Generator):
    items = []
    for cap in sorted((root / "images").glob("*_cap.jpg")):
        sid = cap.name[:-8]
        a, b = load_rgb(cap), load_rgb(root / "images" / f"{sid}_ref.jpg")
        if b.shape != a.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        items.append((f"{sid}/global", squash(b), squash(a)))
        if n_tiles > 0 and len([x for x in items if "/tile" in x[0]]) < n_tiles:
            H, W = a.shape[:2]
            s = min(H, W) // 2
            y, x = int(rng.integers(0, H - s)), int(rng.integers(0, W - s))
            items.append((f"{sid}/tile", squash(b[y:y + s, x:x + s]), squash(a[y:y + s, x:x + s])))
    return items


def synthetic_pairs(root: Path, rng: np.random.Generator, n: int = 4):
    """Paste a patch cut from elsewhere in the same image into the after image (COCO-inpainting-like)."""
    caps = sorted((root / "images").glob("*_cap.jpg"))
    out = []
    for i in range(n):
        base = squash(load_rgb(caps[(i * 7) % len(caps)]))
        src = squash(load_rgb(caps[(i * 7 + 3) % len(caps)]))
        after = base.copy()
        sz = int(rng.integers(40, 80))
        y, x = int(rng.integers(20, 256 - sz - 20)), int(rng.integers(20, 256 - sz - 20))
        sy, sx = int(rng.integers(0, 256 - sz)), int(rng.integers(0, 256 - sz))
        after[y:y + sz, x:x + sz] = src[sy:sy + sz, sx:sx + sz]
        out.append((f"synthetic{i}", base, after, (x, y, x + sz, y + sz)))
    return out


# --------------------------------------------------------------------------- metrics
def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def boxes_of(fused: torch.Tensor, branch: str, k: int = 100):
    c = {"before": (0, slice(2, 4), slice(6, 8)), "after": (1, slice(4, 6), slice(8, 10))}[branch]
    return decode_heatmap(fused[:, c[0]:c[0] + 1], fused[:, c[1]], fused[:, c[2]], k=k)[0]   # (k, 5)


def box_agreement(ref_boxes: torch.Tensor, test_boxes: torch.Tensor, thr: float, iou_thr: float) -> float:
    """Fraction of reference boxes with score >= thr matched by a test box (score >= thr/2) at IoU >= iou_thr."""
    r = ref_boxes[ref_boxes[:, 4] >= thr].tolist()
    t = test_boxes[test_boxes[:, 4] >= thr / 2].tolist()
    if not r:
        return 1.0
    return sum(any(iou(a, b) >= iou_thr for b in t) for a in r) / len(r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--data", default=Path(os.environ.get("CYWS_DATA") or ROOT / "data" / "testCYWS"), type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tiles", type=int, default=8)
    ap.add_argument("--out", default=ROOT / "build/verify", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(args.device)
    rng = np.random.default_rng(0)

    ref = RefModel()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    res = ref.load_state_dict(sd, strict=True)
    print(f"reference strict load OK ({len(sd)} tensors)")
    ref.eval().to(dev)

    unf = cyws_mla.CywsMLA()
    info = unf.load_checkpoint(args.ckpt)
    print(f"rewrite strict load OK ({info['keys']} tensors)")
    unf.eval().to(dev)
    fold = cyws_mla.build(args.ckpt, export=True).to(dev).eval()

    # Co-attention logit ranges (input to the softmax), collected on the folded model.
    logit_stats: dict[int, list[float]] = {0: [], 1: [], 2: []}
    orig_einsum = torch.einsum

    items = dataset_inputs(args.data, args.tiles, rng)
    synth = synthetic_pairs(args.data, rng)
    rows, worst = [], {"unf_heat": 0.0, "unf_whoff": 0.0, "fold_heat": 0.0, "fold_whoff": 0.0}
    peak_mismatch, box_iou_min = 0, 1.0
    bf16_agree, bf16_heat = [], []
    with torch.no_grad():
        for name, b, a, *rest in [(n, b, a) for n, b, a in items] + [(n, b, a, bx) for n, b, a, bx in synth]:
            tb, ta = to_tensor(b).to(dev), to_tensor(a).to(dev)
            r = ref_to_fused(ref(tb, ta))
            u = unf(tb, ta)
            # capture co-attention scores by temporarily wrapping einsum
            level = {"i": 0}
            def spy(eq, *ops):
                outv = orig_einsum(eq, *ops)
                if eq == "nchw,nchq->nqhw":
                    lv = {64: 0, 256: 1, 1024: 2}.get(outv.shape[1], -1)
                    if lv >= 0:
                        logit_stats[lv] += [float(outv.min()), float(outv.max())]
                return outv
            torch.einsum = spy
            try:
                f = fold(tb, ta)
            finally:
                torch.einsum = orig_einsum
            d_unf_h = float((u[:, :2] - r[:, :2]).abs().max())
            # wh/off are pixel quantities up to ~256: compare relative to their magnitude.
            d_unf_w = float((u[:, 2:] - r[:, 2:]).abs().max() / r[:, 2:].abs().max().clamp(min=1.0))
            d_fold_h = float((f[:, :2] - u[:, :2]).abs().max())
            d_fold_w = float((f[:, 2:] - u[:, 2:]).abs().max() / u[:, 2:].abs().max().clamp(min=1.0))
            worst["unf_heat"] = max(worst["unf_heat"], d_unf_h)
            worst["unf_whoff"] = max(worst["unf_whoff"], d_unf_w)
            worst["fold_heat"] = max(worst["fold_heat"], d_fold_h)
            worst["fold_whoff"] = max(worst["fold_whoff"], d_fold_w)
            for br in ("before", "after"):
                bu, bf = boxes_of(u, br), boxes_of(f, br)
                # same strong peaks (score >= 0.1); weak near-tie peaks may reorder in float32
                strong = bu[:, 4] >= 0.1
                if int(strong.sum()) != int((bf[:, 4] >= 0.1).sum()):
                    peak_mismatch += 1
                for x, y in zip(bu[strong].tolist(), bf[strong].tolist()):
                    box_iou_min = min(box_iou_min, iou(x, y))
            # bf16 autocast (early signal for the compiled bf16 model)
            if dev.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    fb16 = fold(tb, ta).float()
                hb = float(torch.nn.functional.cosine_similarity(fb16[:, :2].flatten(), f[:, :2].flatten(), dim=0))
                bf16_heat.append(hb)
                for br in ("before", "after"):
                    bf16_agree.append(box_agreement(boxes_of(f, br), boxes_of(fb16, br), 0.3, 0.7))
            row = {"name": name, "unf_heat": d_unf_h, "unf_whoff": d_unf_w, "fold_heat": d_fold_h,
                   "fold_whoff": d_fold_w, "top_after": [round(v, 3) for v in boxes_of(f, "after")[0].tolist()],
                   "top_before": [round(v, 3) for v in boxes_of(f, "before")[0].tolist()]}
            if rest:
                gt = rest[0]
                ba = boxes_of(f, "after")
                hits = [(float(s[4]), iou(s[:4].tolist(), gt)) for s in ba if s[4] >= 0.3]
                best = max(hits, key=lambda t: t[1], default=(0.0, 0.0))
                row["paste_gt"], row["paste_best_score_iou"] = gt, best
            rows.append(row)

    # Exactness: in float64 on the CPU the rewrite must equal upstream to ~1e-9,
    # which shows the float32 differences above are rounding, not a modelling error.
    ref64, unf64 = ref.double().cpu(), unf.double().cpu()
    exact = 0.0
    with torch.no_grad():
        for _, b, a in items[:3]:
            tb, ta = to_tensor(b).double(), to_tensor(a).double()
            exact = max(exact, float((unf64(tb, ta) - ref_to_fused(ref64(tb, ta))).abs().max()))

    synth_rows = [r for r in rows if "paste_gt" in r]
    synth_ok = sum(1 for r in synth_rows if r["paste_best_score_iou"][1] >= 0.5)
    lstats = {f"level{lv}": [round(min(v), 1), round(max(v), 1)] if v else None for lv, v in logit_stats.items()}
    summary = {
        "inputs": len(rows),
        "worst": worst,
        "float64_max_abs_rewrite_vs_ref": exact,
        "pass_unfolded_vs_ref": exact <= 1e-9 and worst["unf_heat"] <= 1e-5 and worst["unf_whoff"] <= 1e-4,
        "pass_folded_vs_unfolded": worst["fold_heat"] <= 1e-4 and worst["fold_whoff"] <= 1e-3
                                   and box_iou_min > 0.999 and peak_mismatch == 0,
        "strong_peak_count_mismatches": peak_mismatch,
        "min_box_iou_folded_vs_unfolded": box_iou_min,
        "synthetic_paste_detected": f"{synth_ok}/{len(synth_rows)}",
        "coattention_logit_range": lstats,
        "bf16_autocast": {"heat_cosine_min": min(bf16_heat) if bf16_heat else None,
                          "box_agreement_mean": float(np.mean(bf16_agree)) if bf16_agree else None,
                          "box_agreement_min": float(np.min(bf16_agree)) if bf16_agree else None},
    }
    (args.out / "verify_report.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    ok = summary["pass_unfolded_vs_ref"] and summary["pass_folded_vs_unfolded"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
