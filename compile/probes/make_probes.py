#!/usr/bin/env python3
"""Export small real-weight sub-graphs of CYWS that exercise the risky MLA patterns.

Each probe compiles in minutes, so a pattern that falls off the MLA (extra MLA
ELF or an A65 stage) is found before the long full-model compile finishes.

  coam32   co-attention at 32x32 (1024 tokens), both directions   2 x (1,512,32,32)
  coam8    co-attention at 8x8, both directions                    2 x (1,2048,8,8)
  scse16   decoder block 0 attention1 scSE on 6144 ch @ 16x16       (1,6144,16,16)
  dec256   last decoder block (128->64 @ 256) + fused CenterNet head (1,128,128,128)
  stem     conv1 + maxpool on both 3-channel inputs                2 x (1,3,256,256)

Usage (host venv): compile/probes/make_probes.py [--out build/probes]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
import cyws_mla  # noqa: E402
from export_onnx import constants_to_initializers  # noqa: E402


class Coam(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, in0_before, in1_after):
        # One output: the two directions share only the inputs, and separate
        # outputs would make two disconnected sub-graphs (two MLA segments).
        return torch.cat(self.m(in0_before, in1_after), 1)


class Scse(nn.Module):
    def __init__(self, m, part="both"):
        super().__init__()
        self.m, self.part = m, part

    def forward(self, in0_x):
        if self.part == "cse":
            return in0_x * self.m.channel_gate(in0_x)
        if self.part == "sse":
            return in0_x * self.m.sSE(in0_x)
        return self.m(in0_x)


class DecHead(nn.Module):
    def __init__(self, blk, head):
        super().__init__()
        self.blk, self.head = blk, head

    def forward(self, in0_x):
        h, wh, off = self.head(self.blk(in0_x, None))
        return torch.cat([h, wh, off], 1)


class Stem(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, in0_before, in1_after):
        f = lambda x: self.enc.maxpool(self.enc.relu(self.enc.conv1(x)))
        return torch.cat([f(in0_before), f(in1_after)], 1)


def export(mod: nn.Module, inputs: dict[str, tuple], path: Path, out_names: list[str]) -> None:
    args = tuple(torch.rand(s) for s in inputs.values())
    with torch.no_grad():
        torch.onnx.export(mod.eval(), args, str(path), input_names=list(inputs), output_names=out_names,
                          opset_version=17, do_constant_folding=True, dynamo=False)
    m = onnx.load(str(path))
    import onnxsim
    m, ok = onnxsim.simplify(m)
    assert ok
    constants_to_initializers(m)
    onnx.save(m, str(path))
    ops = sorted({n.op_type for n in m.graph.node})
    print(f"{path.name}: inputs {list(inputs.values())} ops {ops}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "build/ckpt/pretrained-resnet50-3x-coam-scSE-affine.ckpt"))
    ap.add_argument("--out", type=Path, default=ROOT / "build/probes")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    m = cyws_mla.build(args.ckpt, export=True)
    dec = m.unet_model.decoder
    export(Coam(m.coattention_modules[2]), {"in0_before": (1, 512, 32, 32), "in1_after": (1, 512, 32, 32)},
           args.out / "coam32.onnx", ["out"])
    export(Coam(m.coattention_modules[0]), {"in0_before": (1, 2048, 8, 8), "in1_after": (1, 2048, 8, 8)},
           args.out / "coam8.onnx", ["out"])
    sc = dec.blocks[0].attention1.attention
    export(Scse(sc), {"in0_x": (1, 6144, 16, 16)}, args.out / "scse16.onnx", ["out"])
    # scSE bisection variants
    export(Scse(sc, "cse"), {"in0_x": (1, 6144, 16, 16)}, args.out / "scse16_cse.onnx", ["out"])
    export(Scse(sc, "sse"), {"in0_x": (1, 6144, 16, 16)}, args.out / "scse16_sse.onnx", ["out"])
    sc.pool = "mean"
    export(Scse(sc), {"in0_x": (1, 6144, 16, 16)}, args.out / "scse16_mean.onnx", ["out"])
    sc.pool, sc.expand = "gap", "conv"
    export(Scse(sc), {"in0_x": (1, 6144, 16, 16)}, args.out / "scse16_conv.onnx", ["out"])
    sc.expand = "broadcast"
    export(Scse(dec.blocks[2].attention1.attention), {"in0_x": (1, 512, 64, 64)}, args.out / "scse64.onnx", ["out"])
    export(DecHead(dec.blocks[4], m.centernet_head), {"in0_x": (1, 128, 128, 128)},
           args.out / "dec256.onnx", ["out"])
    export(Stem(m.unet_model.encoder), {"in0_before": (1, 3, 256, 256), "in1_after": (1, 3, 256, 256)},
           args.out / "stem.onnx", ["out"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
