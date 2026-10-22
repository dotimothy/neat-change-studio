"""CYWS ("The Change You Want to See", Sachdeva & Zisserman, WACV 2023) for the SiMa MLA.

A dependency-free re-implementation of upstream's CenterNetWithCoAttention
(ResNet-50 Siamese U-Net + co-attention + mmdet CenterNetHead). Module names
mirror the upstream Lightning checkpoint exactly, so it loads with strict=True:
any missing or unexpected key is an error, unlike upstream's own loader, which
silently drops mismatched weights.

The forward pass only uses ops that the SiMa compiler places on the MLA:
    Conv, Relu, Sigmoid, Add, Mul, Concat, Split, Einsum, Softmax, MaxPool,
    Resize (nearest x2), GlobalAveragePool.
In particular:
  * BatchNorm is folded into the preceding conv (fold_batchnorm()).
  * Co-attention puts tokens on the W axis, (1, C, 1, T), flattening with
    Split+Concat instead of Reshape, and uses the same Einsum/Split-Concat
    attention pattern that compiles to one MLA ELF:
        S = Einsum("nchw,nchq->nqhw", Q, K)      (1, Tk, 1, Tq)
        P = Softmax(S, axis=1)                   softmax over reference tokens
        O = Einsum("nchw,nqhc->nqhw", P, V)      (1, C, 1, Tq)
    CYWS has no 1/sqrt(d) scale, so none is added.
  * The three CenterNet branches are fused into one 3x3 conv (64->192) and a
    block-diagonal 1x1 conv (192->5); this is exact.

Inputs: in0_before, in1_after, each (1, 3, 256, 256) float32 RGB / 255 (upstream
does no mean/std normalisation). Upstream's "left" image is before, "right" is after.
Outputs ("fused"): out (1, 10, 256, 256) =
    [heat_b, heat_a, wh_b(w,h), wh_a(w,h), off_b(x,y), off_a(x,y)]
("split"): out0_heat (1,2,..), out1_wh (1,4,..), out2_off (1,4,..), same channel order.
License: Apache-2.0. An independent reimplementation; CYWS itself is Apache-2.0
(confirmed by its authors, see NOTICE).
"""

from __future__ import annotations

from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_NAMES = ("in0_before", "in1_after")
FUSED_OUTPUT_NAMES = ("out",)
SPLIT_OUTPUT_NAMES = ("out0_heat", "out1_wh", "out2_off")


# --------------------------------------------------------------------------- encoder
class Bottleneck(nn.Module):
    """torchvision ResNet-50 v1.5 bottleneck (stride on the 3x3 conv)."""

    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idt = x if self.downsample is None else self.downsample(x)
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        return self.relu(y + idt)


def _make_layer(inplanes: int, planes: int, blocks: int, stride: int) -> nn.Sequential:
    down = None
    if stride != 1 or inplanes != planes * 4:
        down = nn.Sequential(nn.Conv2d(inplanes, planes * 4, 1, stride=stride, bias=False),
                             nn.BatchNorm2d(planes * 4))
    layers = [Bottleneck(inplanes, planes, stride, down)]
    layers += [Bottleneck(planes * 4, planes) for _ in range(blocks - 1)]
    return nn.Sequential(*layers)


class ResNet50Encoder(nn.Module):
    """smp ResNetEncoder("resnet50"): returns the 6 feature maps [x, stem, l1, l2, l3, l4]."""

    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = _make_layer(64, 64, 3, 1)
        self.layer2 = _make_layer(256, 128, 4, 2)
        self.layer3 = _make_layer(512, 256, 6, 2)
        self.layer4 = _make_layer(1024, 512, 3, 2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f0 = x
        f1 = self.relu(self.bn1(self.conv1(x)))           # 64 @ 128
        f2 = self.layer1(self.maxpool(f1))                 # 256 @ 64
        f3 = self.layer2(f2)                               # 512 @ 32
        f4 = self.layer3(f3)                               # 1024 @ 16
        f5 = self.layer4(f4)                               # 2048 @ 8
        return [f0, f1, f2, f3, f4, f5]


# --------------------------------------------------------------------------- decoder
class SCSE(nn.Module):
    """smp SCSEModule: x * cSE(x) + x * sSE(x). Names match cSE.{1,3} / sSE.0."""

    def __init__(self, ch: int, reduction: int = 16, expand: Literal["broadcast", "conv"] = "broadcast",
                 pool: Literal["gap", "mean"] = "gap"):
        super().__init__()
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch, ch // reduction, 1),
                                 nn.ReLU(inplace=False), nn.Conv2d(ch // reduction, ch, 1), nn.Sigmoid())
        self.sSE = nn.Sequential(nn.Conv2d(ch, 1, 1), nn.Sigmoid())
        self.expand = expand
        self.pool = pool
        self.ch = ch

    def channel_gate(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool == "mean":   # ReduceMean over H,W instead of GlobalAveragePool
            g = x.mean(dim=(2, 3), keepdim=True)
            return self.cSE[4](self.cSE[3](self.cSE[2](self.cSE[1](g))))
        return self.cSE(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.sSE(x)
        if self.expand == "conv":
            # Contingency for compilers that reject a (1,1,H,W) broadcast: a fixed
            # 1->C conv of ones copies the spatial gate to every channel.
            s = F.conv2d(s, torch.ones(self.ch, 1, 1, 1, dtype=s.dtype, device=s.device))
        return x * self.channel_gate(x) + x * s


class Attention(nn.Module):
    """smp md.Attention wrapper (keeps the `.attention.` level in parameter names)."""

    def __init__(self, ch: int | None, expand: str = "broadcast"):
        super().__init__()
        self.attention = SCSE(ch, expand=expand) if ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(x)


def _conv_bn_relu(cin: int, cout: int) -> nn.Sequential:
    """smp Conv2dReLU: Sequential(conv(bias=False), BatchNorm, ReLU) -> names .0/.1."""
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=False))


class DecoderBlock(nn.Module):
    def __init__(self, cin: int, skip: int, cout: int, expand: str = "broadcast"):
        super().__init__()
        # Upstream disables attention for the skip-less last block (attention_type=None).
        att = skip > 0
        self.conv1 = _conv_bn_relu(cin + skip, cout)
        self.attention1 = Attention(cin + skip if att else None, expand)
        self.conv2 = _conv_bn_relu(cout, cout)
        self.attention2 = Attention(cout if att else None, expand)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        return self.attention2(self.conv2(self.conv1(x)))


class UnetDecoder(nn.Module):
    """Forked smp UnetDecoder with num_coam_layers=3 and decoder_channels=(256,256,128,128,64)."""

    IN = (4096, 256, 256, 128, 128)     # head 2048 doubled by co-attention
    SKIP = (2048, 1024, 256, 64, 0)     # first two skips doubled by co-attention
    OUT = (256, 256, 128, 128, 64)

    def __init__(self, expand: str = "broadcast"):
        super().__init__()
        self.center = nn.Identity()
        self.blocks = nn.ModuleList(DecoderBlock(i, s, o, expand) for i, s, o in zip(self.IN, self.SKIP, self.OUT))

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        f = list(feats[1:])[::-1]           # [l4, l3, l2, l1, stem]
        x = self.center(f[0])
        skips = f[1:]
        for i, blk in enumerate(self.blocks):
            x = blk(x, skips[i] if i < len(skips) else None)
        return x


class UnetModel(nn.Module):
    def __init__(self, in_ch: int = 3, expand: str = "broadcast"):
        super().__init__()
        self.encoder = ResNet50Encoder(in_ch)
        self.decoder = UnetDecoder(expand)


# --------------------------------------------------------------------------- co-attention
def _flatten_hw(x: torch.Tensor) -> torch.Tensor:
    """(1,C,H,W) -> (1,C,1,H*W) row-major, via Split(axis=2) + Concat(axis=3) (no Reshape)."""
    if x.shape[2] == 1:
        return x
    return torch.cat(torch.split(x, 1, dim=2), dim=3)


def _unflatten_hw(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """(1,C,1,H*W) -> (1,C,H,W), via Split(axis=3) + Concat(axis=2)."""
    if h == 1:
        return x
    return torch.cat(torch.split(x, w, dim=3), dim=2)


class CoAttentionLayer(nn.Module):
    """Upstream CoAttentionLayer, same parameter names, MLA-friendly forward.

    Upstream: A = softmax_p(einsum("bcij,bckl->bijkl", Q, K) flattened over (k,l)),
              out = einsum("bijp,bcp->bcij", A, V), V = reference features (unprojected).
    """

    def __init__(self, cin: int, hidden: int):
        super().__init__()
        self.reference_dimensionality_reduction = nn.Conv2d(cin, hidden, 1, bias=True)
        self.query_dimensionality_reduction = nn.Conv2d(cin, hidden, 1, bias=True)
        self.query_chunks = 1   # >1 splits the query tokens (exact; smaller score tensors)

    def forward(self, query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        _, _, h, w = query.shape
        q = _flatten_hw(self.query_dimensionality_reduction(query))       # (1, c, 1, Tq)
        k = _flatten_hw(self.reference_dimensionality_reduction(reference))  # (1, c, 1, Tk)
        v = _flatten_hw(reference)                                         # (1, C, 1, Tk)
        qs = torch.chunk(q, self.query_chunks, dim=3) if self.query_chunks > 1 else (q,)
        outs = []
        for qc in qs:
            s = torch.einsum("nchw,nchq->nqhw", qc, k)     # (1, Tk, 1, Tq_chunk)
            p = torch.softmax(s, dim=1)                    # over reference tokens
            outs.append(torch.einsum("nchw,nqhc->nqhw", p, v))   # (1, C, 1, Tq_chunk)
        o = outs[0] if len(outs) == 1 else torch.cat(outs, dim=3)
        return _unflatten_hw(o, h, w)


class CoAttentionModule(nn.Module):
    def __init__(self, cin: int, hidden: int):
        super().__init__()
        self.attention_layer = CoAttentionLayer(cin, hidden)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weighted_r = self.attention_layer(left, right)   # left queries the right image
        weighted_l = self.attention_layer(right, left)
        return torch.cat([left, weighted_r], dim=1), torch.cat([right, weighted_l], dim=1)


# --------------------------------------------------------------------------- head
class CenterNetHead(nn.Module):
    """mmdet 2.x CenterNetHead(in_channel=64, feat_channel=64, num_classes=1), forward only."""

    def __init__(self, cin: int = 64, feat: int = 64):
        super().__init__()
        def branch(n: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(cin, feat, 3, padding=1), nn.ReLU(inplace=False), nn.Conv2d(feat, n, 1))
        self.heatmap_head = branch(1)
        self.wh_head = branch(2)
        self.offset_head = branch(2)
        self.fused_conv3: nn.Conv2d | None = None   # set by fuse(); registered submodules
        self.fused_conv1: nn.Conv2d | None = None

    def fuse(self) -> None:
        """Merge the three branches into conv3x3 64->192 + block-diagonal conv1x1 192->5 (exact)."""
        heads = (self.heatmap_head, self.wh_head, self.offset_head)
        c3 = nn.Conv2d(heads[0][0].in_channels, sum(h[0].out_channels for h in heads), 3, padding=1)
        c1 = nn.Conv2d(c3.out_channels, 5, 1)
        with torch.no_grad():
            c3.weight.copy_(torch.cat([h[0].weight for h in heads], 0))
            c3.bias.copy_(torch.cat([h[0].bias for h in heads], 0))
            c1.weight.zero_()
            o_in, o_out = 0, 0
            for h in heads:
                fi, fo = h[2].in_channels, h[2].out_channels
                c1.weight[o_out:o_out + fo, o_in:o_in + fi] = h[2].weight
                c1.bias[o_out:o_out + fo] = h[2].bias
                o_in += fi; o_out += fo
        self.fused_conv3 = c3.to(heads[0][0].weight)
        self.fused_conv1 = c1.to(heads[0][0].weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.fused_conv3 is not None:
            y = self.fused_conv1(F.relu(self.fused_conv3(x)))
            heat, wh, off = torch.split(y, [1, 2, 2], dim=1)
            return torch.sigmoid(heat), wh, off
        return torch.sigmoid(self.heatmap_head(x)), self.wh_head(x), self.offset_head(x)


# --------------------------------------------------------------------------- full model
class CywsMLA(nn.Module):
    """CYWS with upstream parameter names and an MLA-friendly forward.

    Args:
        outputs: "fused" -> one (1,10,256,256) output; "split" -> heat/wh/off outputs.
        query_chunks: per co-attention level (8x8, 16x16, 32x32) query split count.
        scse_expand: "broadcast" (x * sSE) or "conv" (sSE expanded by a ones conv). The SiMa
            compiler fails on scSE with two broadcast multiplies; "conv" compiles.
        max_conv_params: >0 splits convs with more weights than this along input channels.
        in_ch: 3, or 16 to zero-pad conv1 (contingency for 3-channel tessellation asserts);
            the caller then feeds 16-channel inputs whose extra channels are zero.
    """

    def __init__(self, outputs: Literal["fused", "split"] = "fused",
                 query_chunks: Sequence[int] = (1, 1, 1),
                 scse_expand: Literal["broadcast", "conv"] = "broadcast", in_ch: int = 3,
                 max_conv_params: int = 0):
        super().__init__()
        self.unet_model = UnetModel(3, scse_expand)
        self.coattention_modules = nn.ModuleList(
            CoAttentionModule(c, h) for c, h in ((2048, 256), (1024, 128), (512, 64)))
        self.centernet_head = CenterNetHead()
        self.outputs = outputs
        self.in_ch = in_ch
        self.max_conv_params = int(max_conv_params)
        self.split_convs: list[str] = []
        for m, n in zip(self.coattention_modules, query_chunks):
            m.attention_layer.query_chunks = int(n)

    def branch_features(self, before: torch.Tensor, after: torch.Tensor) -> tuple[list, list]:
        fb = self.unet_model.encoder(before)
        fa = self.unet_model.encoder(after)
        for i, coam in enumerate(self.coattention_modules):
            fb[-(i + 1)], fa[-(i + 1)] = coam(fb[-(i + 1)], fa[-(i + 1)])
        return fb, fa

    def forward(self, in0_before: torch.Tensor, in1_after: torch.Tensor):
        fb, fa = self.branch_features(in0_before, in1_after)
        hb, wb, ob = self.centernet_head(self.unet_model.decoder(fb))
        ha, wa, oa = self.centernet_head(self.unet_model.decoder(fa))
        heat = torch.cat([hb, ha], 1)
        wh = torch.cat([wb, wa], 1)
        off = torch.cat([ob, oa], 1)
        if self.outputs == "split":
            return heat, wh, off
        return torch.cat([heat, wh, off], 1)

    # -- weight handling ---------------------------------------------------------
    def load_checkpoint(self, path: str) -> dict:
        """Strictly load an upstream Lightning checkpoint (raises on any key mismatch)."""
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        sd = {k: v for k, v in sd.items() if not k.endswith("num_batches_tracked")}
        own = {k for k in self.state_dict() if not k.endswith("num_batches_tracked")}
        missing, unexpected = sorted(own - set(sd)), sorted(set(sd) - own)
        if missing or unexpected:
            raise KeyError(f"checkpoint mismatch: missing={missing[:10]} (+{max(0, len(missing)-10)}), "
                           f"unexpected={unexpected[:10]} (+{max(0, len(unexpected)-10)})")
        bad = [k for k, v in sd.items() if v.shape != self.state_dict()[k].shape]
        if bad:
            raise ValueError(f"shape mismatch for {bad[:10]}")
        self.load_state_dict(sd, strict=False)   # num_batches_tracked buffers are the only gap
        return {"keys": len(sd)}

    def prepare_for_export(self) -> "CywsMLA":
        """Fold BatchNorm, fuse the head, optionally pad conv1. Call after load_checkpoint()."""
        self.eval()
        fold_batchnorm(self)
        self.centernet_head.fuse()
        if self.max_conv_params > 0:
            self.split_convs = split_large_convs(self, self.max_conv_params)
        if self.in_ch != 3:
            enc = self.unet_model.encoder
            w = enc.conv1.weight
            conv = nn.Conv2d(self.in_ch, w.shape[0], 7, stride=2, padding=3, bias=enc.conv1.bias is not None)
            with torch.no_grad():
                conv.weight.zero_()
                conv.weight[:, :3] = w
                if enc.conv1.bias is not None:
                    conv.bias.copy_(enc.conv1.bias)
            enc.conv1 = conv
        return self


# --------------------------------------------------------------------------- large convs
class SplitConv(nn.Module):
    """conv(x) computed as sum_i conv_i(x_i) over input-channel chunks (Split -> Conv*n -> Add).

    Exact up to float summation order. Used for convs whose weight tensor is too
    large for the MLA compiler's layer memory (decoder block 0: 3x3, 6144->256).
    """

    def __init__(self, conv: nn.Conv2d, chunks: int):
        super().__init__()
        cin = conv.in_channels
        if cin % chunks:
            raise ValueError(f"{cin} input channels do not split into {chunks} chunks")
        self.sizes = [cin // chunks] * chunks
        self.convs = nn.ModuleList()
        with torch.no_grad():
            for i, w in enumerate(torch.split(conv.weight, self.sizes, dim=1)):
                c = nn.Conv2d(w.shape[1], conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                              conv.dilation, bias=(i == 0 and conv.bias is not None))
                c.weight.copy_(w)
                if c.bias is not None:
                    c.bias.copy_(conv.bias)
                self.convs.append(c.to(conv.weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = torch.split(x, self.sizes, dim=1)
        y = self.convs[0](parts[0])
        for c, xi in zip(self.convs[1:], parts[1:]):
            y = y + c(xi)
        return y


def split_large_convs(model: nn.Module, max_params: int) -> list[str]:
    """Replace every Conv2d with more than max_params weights by a SplitConv; returns their names."""
    done = []
    for name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Conv2d) and child.weight.numel() > max_params and child.groups == 1:
                chunks = 2
                while child.weight.numel() / chunks > max_params or child.in_channels % chunks:
                    chunks += 1
                    if chunks > child.in_channels:
                        raise ValueError(f"cannot split {name}.{child_name}")
                setattr(parent, child_name, SplitConv(child, chunks))
                done.append(f"{name}.{child_name} -> {chunks} chunks")
    return done


# --------------------------------------------------------------------------- BN folding
def _fold(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Return a conv equal to bn(conv(x)) in eval mode, computed in float64."""
    w = conv.weight.detach().double()
    b = conv.bias.detach().double() if conv.bias is not None else torch.zeros(w.shape[0], dtype=torch.float64)
    g, beta = bn.weight.detach().double(), bn.bias.detach().double()
    mu, var = bn.running_mean.detach().double(), bn.running_var.detach().double()
    s = g / torch.sqrt(var + bn.eps)
    out = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                    conv.padding, conv.dilation, conv.groups, bias=True)
    with torch.no_grad():
        out.weight.copy_((w * s[:, None, None, None]).float())
        out.bias.copy_((beta + (b - mu) * s).float())
    return out


def fold_batchnorm(model: nn.Module) -> None:
    """Fold every BatchNorm2d into the conv right before it (encoder, decoder Conv2dReLU)."""
    for m in list(model.modules()):   # snapshot: children are replaced while walking
        if isinstance(m, Bottleneck):
            m.conv1, m.bn1 = _fold(m.conv1, m.bn1), nn.Identity()
            m.conv2, m.bn2 = _fold(m.conv2, m.bn2), nn.Identity()
            m.conv3, m.bn3 = _fold(m.conv3, m.bn3), nn.Identity()
            if m.downsample is not None:
                m.downsample = nn.Sequential(_fold(m.downsample[0], m.downsample[1]))
        elif isinstance(m, ResNet50Encoder):
            m.conv1, m.bn1 = _fold(m.conv1, m.bn1), nn.Identity()
        elif isinstance(m, nn.Sequential) and len(m) == 3 and isinstance(m[0], nn.Conv2d) \
                and isinstance(m[1], nn.BatchNorm2d):
            m[0], m[1] = _fold(m[0], m[1]), nn.Identity()
    left = [n for n, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)]
    if left:
        raise RuntimeError(f"unfolded BatchNorm left: {left[:5]}")


def build(ckpt: str, *, export: bool = True, **kw) -> CywsMLA:
    """Load the upstream checkpoint strictly; with export=True also fold/fuse for ONNX."""
    m = CywsMLA(**kw)
    m.load_checkpoint(ckpt)
    m.eval()
    return m.prepare_for_export() if export else m
