#!/usr/bin/env python3
"""Publish compiled change-detection models to Hugging Face, for the web demo's model catalog.

Each model is one MPK, uploaded into a folder of its name. `models.json` at the top of the
repository lists them with sizes and SHA-256 sums, and the web app reads that list to offer
the models for download (webapp/server.py, `Hub`): it fetches a model onto the board, checks
its sum, and can then load it on the MLA. The repository also gets a model card (README.md),
and is added to a collection.

There is one repository per model family:
    cyws    dotimothy/sima-cyws     CYWS, trained for change detection (two 256x256 windows in)
    dinov2  dotimothy/sima-dinov2   DINOv2 ViT-S/14, used zero-shot (one 448x448 image in)
    c3po    dotimothy/sima-c3po     C-3PO change segmentation, PCD and ChangeSim checkpoints (two 256x256 images in)

    python3 tools/publish_hub.py                              # CYWS: both models, card, list, collection
    python3 tools/publish_hub.py --only w8a16                 # one model (the list keeps the others)
    python3 tools/publish_hub.py --list-only                  # models.json and the card only
    python3 tools/publish_hub.py --family dinov2              # DINOv2 (build in ../DINOv2/build/bf16_448/)
    python3 tools/publish_hub.py --family dinov2 --dry-run out/hub   # write models.json and README.md, upload nothing
    python3 tools/publish_hub.py --family c3po --dry-run out/hub     # C-3PO (builds in ../C3PO/build/bf16_*_256/)

An upload needs `huggingface_hub` and a login with write access (`hf auth login`); a dry run
needs neither.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = "https://github.com/dotimothy/neat-change-studio"
COLLECTION = "SiMa.ai Change Detection Models"
COLLECTION_ABOUT = ("Change detection compiled for SiMa.ai Modalix: CYWS (trained) and zero-shot DINOv2. "
                    "Two photos in, boxes around what changed out.")
assert len(COLLECTION_ABOUT) <= 150, "Hugging Face caps a collection's description at 150 characters"

# ---------------------------------------------------------------------------- CYWS
REPO = "dotimothy/sima-cyws"
UPSTREAM = "https://github.com/ragavsachdeva/The-Change-You-Want-to-See"
NOTE = "CYWS (WACV 2023), bf16 and w8a16, one MLA stage. ~35 ms per pair. Apache-2.0."

# name on the hub (and in the app) -> build file, and what the catalog says about it.
# Numbers are from report.md: Model.benchmark on a Modalix DevKit (NEAT 0.5.0), and the app's
# balanced preset on the testCYWS set (28 phone-photo pairs, 47 annotated changes).
MODELS = {
    "bf16": {
        "build": "build/bf16/cyws_bf16_mpk.tar.gz", "title": "CYWS bf16",
        "precision": "BF16",
        "about": "The most faithful build: matches the FP32 model's boxes one for one. The default.",
        "latency_ms": 35.2, "fps": 36, "watts": 13.2,
        "accuracy": {"changes_found": "33 / 47", "pixel_f1": 0.587, "false_alarm_pairs": "2 / 15",
                     "box_agreement_vs_fp32": 1.0, "heat_cosine_min": 0.994},
    },
    "w8a16": {
        "build": "build/w8a16_minmax/cyws_w8a16_mpk.tar.gz", "title": "CYWS w8a16",
        "precision": "A_BF16_W_INT8",
        "about": "Int8 weights with bf16 activations: 40% smaller and ~23% faster per window, slightly less exact.",
        "latency_ms": 27.2, "fps": None, "watts": None,
        "accuracy": {"changes_found": "31 / 47", "pixel_f1": 0.581, "false_alarm_pairs": "3 / 15",
                     "box_agreement_vs_fp32": 0.966, "heat_cosine_min": 0.826},
    },
}
CARD = {"model": "The Change You Want to See (Sachdeva & Zisserman, WACV 2023)",
        "architecture": "Siamese ResNet-50 U-Net, co-attention, CenterNet head; 49.5M parameters",
        "input": "before + after, 256x256 RGB each", "upstream": "ragavsachdeva/The-Change-You-Want-to-See",
        "license": "Apache-2.0"}

# ---------------------------------------------------------------------------- DINOv2
DINO_REPO = "dotimothy/sima-dinov2"
DINO_UPSTREAM = "https://github.com/facebookresearch/dinov2"
DINO_NOTE = "DINOv2 ViT-S/14 (Meta), bf16, one MLA stage, used zero-shot for change detection. 24.5 ms per image. Apache-2.0."
# Numbers are from ../DINOv2/README.md: Model.benchmark on a Modalix DevKit (NEAT 0.5.0), and
# zero-shot change detection on testCYWS, scored with Change Studio's own metrics.
DINO_MODELS = {
    "dinov2-448": {
        "build": "../DINOv2/build/bf16_448/dinov2_vits14_reg_448_bf16_mpk.tar.gz", "title": "DINOv2 448",
        "precision": "BF16",
        "about": "Zero-shot: finds what changed by comparing DINOv2 features of the two photos, with no change-detection training.",
        "latency_ms": 24.5,                                  # per image, not per pair
        "input": {"size": 448, "images": 1},
        "accuracy": {"changes_found": "28 / 47", "pixel_f1": 0.614, "false_alarm_pairs": "2 / 15"},
    },
}
DINO_CARD = {"model": "DINOv2 ViT-S/14 with registers (Meta)",
             "architecture": "Vision Transformer ViT-S/14 with 4 registers; used zero-shot, no change-detection training",
             "input": "one 448x448 RGB image -> 32x32 patches", "upstream": "facebookresearch/dinov2",
             "license": "Apache-2.0"}

# ---------------------------------------------------------------------------- C-3PO
C3PO_REPO = "dotimothy/sima-c3po"
C3PO_UPSTREAM = "https://github.com/DoctorKey/C-3PO"
C3PO_NOTE = "C-3PO change segmentation (Pattern Recognition 2023), PCD and ChangeSim checkpoints, bf16, one MLA stage. 46.7 ms per pair. MIT."
# Numbers are from ../C3PO/README.md: Model.benchmark on a Modalix DevKit, and the app's c3po
# presets on testCYWS with the FP32 ONNX (bf16 agrees with FP32 on 99.98% of mask pixels).
# The balanced preset (whole photo + gated 0.6 tiles, thr 0.20), like the CYWS and DINOv2
# headlines; all tiles in `accurate`, whole photo only in `whole_photo`.
ACC_PCD = {"changes_found": "28 / 47", "pixel_f1": 0.518, "false_alarm_pairs": "2 / 15",
           "accurate": {"changes_found": "30 / 47", "pixel_f1": 0.554, "false_alarm_pairs": "2 / 15"},
           "whole_photo": {"changes_found": "26 / 47", "pixel_f1": 0.499, "false_alarm_pairs": "1 / 15"}}
ACC_CSB = {"changes_found": "19 / 47", "pixel_f1": 0.524, "false_alarm_pairs": "2 / 15",
           "accurate": {"changes_found": "22 / 47", "pixel_f1": 0.536, "false_alarm_pairs": "2 / 15"},
           "whole_photo": {"changes_found": "19 / 47", "pixel_f1": 0.524, "false_alarm_pairs": "2 / 15"}}
C3PO_MODELS = {
    "c3po-pcd": {
        "build": "../C3PO/build/bf16_pcd_256/c3po_pcd_256_bf16_mpk.tar.gz", "title": "C-3PO PCD",
        "precision": "BF16",
        "about": "Trained on PCD street panoramas: segments what changed between the two photos.",
        "latency_ms": 46.7,                                  # per before/after pair
        "input": {"size": 256, "images": 2},
        "accuracy": ACC_PCD,
    },
    "c3po-csb": {
        "build": "../C3PO/build/bf16_csb_256/c3po_csb_256_bf16_mpk.tar.gz", "title": "C-3PO ChangeSim",
        "precision": "BF16",
        "about": "Trained on ChangeSim (simulated indoor factory, binary): segments what changed between the two photos.",
        "latency_ms": 46.7,
        "input": {"size": 256, "images": 2},
        "accuracy": ACC_CSB,
    },
}
C3PO_CARD = {"model": "C-3PO: How to Reduce Change Detection to Semantic Segmentation (Wang et al., Pattern Recognition 2023)",
             "architecture": "ResNet-18 shared by both photos, MTF (appeared/disappeared/exchanged), MSF over 4 scales, DeepLabV3 head; 44.7M parameters",
             "input": "before + after, 256x256 RGB each", "upstream": "DoctorKey/C-3PO", "license": "MIT"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


def entry(name: str, spec: dict, family: str) -> dict:
    """One models.json entry (the rules it must pass are in webapp/server.py, `Hub`)."""
    path = ROOT / spec["build"]
    out = {"title": spec["title"], "about": spec["about"], "path": name, "family": family,
           "precision": spec["precision"], "latency_ms": spec["latency_ms"], "accuracy": spec["accuracy"],
           "card": FAMILIES[family]["card"],
           "files": [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}]}
    if "input" in spec:
        out["input"] = spec["input"]
    return out


def cyws_card(repo: str, models: dict) -> str:
    def mb(e):
        return f"{sum(f['bytes'] for f in e['files']) / 1e6:.0f} MB"
    rows = "\n".join(
        f"| `{n}/{e['files'][0]['name']}` | {e['precision']} | {e['latency_ms']:g} ms | {e['accuracy']['changes_found']} | "
        f"{e['accuracy']['pixel_f1']:.3f} | {e['accuracy']['false_alarm_pairs']} | {e['accuracy']['box_agreement_vs_fp32']:.3f} | {mb(e)} |"
        for n, e in models.items())
    return f"""---
license: apache-2.0
library_name: sima-neat
pipeline_tag: object-detection
tags:
- sima.ai
- modalix
- mla
- edge
- change-detection
- cyws
---

# CYWS change detection, compiled for the SiMa.ai MLSoC Modalix

[The Change You Want to See]({UPSTREAM}) (CYWS; Sachdeva & Zisserman, WACV 2023) takes a
**before** and an **after** photo of the same scene and returns boxes around what changed:
objects that appeared, disappeared or changed. It is a Siamese ResNet-50 U-Net with
co-attention between the two images and a CenterNet head on each branch.

These are its pretrained weights (`pretrained-resnet50-3x-coam-scSE-affine`) compiled for the
Modalix MLA with SiMa.ai's Model Compiler. The whole network is **one MLA stage, with nothing
on the CPU**, and one before/after pair of 256x256 windows takes about 35 ms (bf16).

| file | precision | MLA latency per window | changes found | pixel F1 | false-alarm pairs | same boxes as FP32 | size |
|---|---|---|---|---|---|---|---|
{rows}

- **Latency:** `Model.benchmark` on a Modalix DevKit (NEAT 0.5.0), plugins included. The bf16
  model runs 36 windows/s at 13.2 W.
- **Changes found, pixel F1, false alarms:** the app's balanced preset (a global window plus
  gated 256-input tiles) on testCYWS: 28 before/after pairs of 12 MP phone photos, with 47
  annotated changes, and 15 pairs where nothing changed. The settings were tuned on the same
  pairs, so treat these numbers as optimistic for new scenes.
- **Same boxes as FP32:** box agreement with the ONNX FP32 model on 97 inputs.

## Input and output

Two inputs, pushed together and in this order:
- `in0_before` and `in1_after`, each float32 256x256x3, HWC, RGB, pixel / 255, with no
  mean/std normalisation.

One fused output, float32 `(10, 256, 256)`. Its channels are `heat_b, heat_a, wh_b(w,h),
wh_a(w,h), off_b(x,y), off_a(x,y)`, with the heat channels already passed through a sigmoid.
- **Decoding:** take 3x3 peaks of each heat map above about 0.29, then build the boxes from
  `wh` and `off` (CenterNet style).
- **Labels:** a box on the after map only is *appeared*, a box on the before map only is
  *disappeared*, and a box on both is *changed*.

## Using them

On a Modalix board, use [Neat Change Studio]({APP}).
Its web demo lists the models in this repository (`models.json`) and downloads one onto the
board from the Models window (Add Model). From a terminal:

```bash
webapp/run.sh --list-hub          # what is here and what is on the board
webapp/run.sh --fetch bf16        # or w8a16, or all: downloaded into models/ and checked
```

Without the app:

```bash
hf download {repo} bf16/cyws_bf16_mpk.tar.gz --local-dir .
```

```python
import numpy as np, pyneat

opt = pyneat.ModelOptions()
opt.preprocess.kind = pyneat.InputKind.Tensor               # raw tensors in, no image preprocessing
model = pyneat.Model("bf16/cyws_bf16_mpk.tar.gz", opt)

def tensor(hwc):
    t = pyneat.Tensor.from_numpy(np.ascontiguousarray(hwc, np.float32), copy=True)
    t.layout = pyneat.TensorLayout.HWC
    return t

before = after = np.zeros((256, 256, 3), np.float32)        # your two crops: RGB / 255, HWC
run = model.build([tensor(before), tensor(after)])
run.push_tensors([tensor(before), tensor(after)])
sample = run.pull(5000)
t = sample.tensors[0] if sample.tensors else sample.tensor
out = np.array(t.to_numpy(), np.float32, copy=True).transpose(2, 0, 1)   # (10, 256, 256)
run.close()
```

Photos larger than 256x256 should be tiled: run one global window, then 256-input tiles where
the global heat map saw a change, so small objects are seen at the scale CYWS was trained on.

These files only run on the MLA. For PyTorch or ONNX Runtime, use the
[original checkpoint]({UPSTREAM}).

## License and attribution

**Apache-2.0.** CYWS is by Ragav Sachdeva and Andrew Zisserman (The Change You Want to See,
WACV 2023, {UPSTREAM}). The authors have confirmed to the maintainer (2026) that CYWS, its
code and pretrained weights, is released under the Apache License 2.0. This supersedes the
CC BY-NC default mentioned in upstream issue #11 (2024). The upstream repository does not
include a LICENSE file yet.

These files are derived from those weights and are Apache-2.0 too:
- BatchNorm is folded into the convolutions;
- the network is rewritten to compile for the MLA, with the same function;
- it is compiled in bf16, or with int8 weights for w8a16.

Nothing was retrained. The code that does this is at
[dotimothy/neat-change-studio]({APP}).

```bibtex
@inproceedings{{sachdeva2023change,
  title     = {{The Change You Want to See}},
  author    = {{Sachdeva, Ragav and Zisserman, Andrew}},
  booktitle = {{Winter Conference on Applications of Computer Vision (WACV)}},
  year      = {{2023}}
}}
```
"""


def dinov2_card(repo: str, models: dict) -> str:
    def mb(e):
        return f"{sum(f['bytes'] for f in e['files']) / 1e6:.0f} MB"
    rows = "\n".join(
        f"| `{n}/{e['files'][0]['name']}` | {e['precision']} | {e['input']['size']} x {e['input']['size']} | "
        f"{e['latency_ms']:g} ms | {mb(e)} |"
        for n, e in models.items())
    return f"""---
license: apache-2.0
library_name: sima-neat
pipeline_tag: image-feature-extraction
tags:
- sima.ai
- modalix
- mla
- edge
- dinov2
- zero-shot
- change-detection
---

# DINOv2 for zero-shot change detection, compiled for the SiMa.ai MLSoC Modalix

[DINOv2]({DINO_UPSTREAM}) (Meta) is a general image feature extractor. This is its ViT-S/14
with 4 registers (`dinov2_vits14_reg4_pretrain`), compiled for the Modalix MLA with SiMa.ai's
Model Compiler. The whole network is **one MLA stage, with nothing on the CPU** (4 EV74 plugins
for the data path), and one 448x448 image takes 24.5 ms.

[Neat Change Studio]({APP}) uses it **zero-shot** to find what changed
between a before and an after photo: nothing was trained for change detection.

| file | precision | input | MLA latency per image | size |
|---|---|---|---|---|
{rows}

- **Latency:** `Model.benchmark` on a Modalix DevKit (NEAT 0.5.0), plugins included: 50 images/s
  at 14.0 W. A before/after pair is two images.
- **Features vs PyTorch FP32** (12 real photos): patch cosine 0.994, centred patch cosine 0.997.

## How it finds changes

Both photos go through DINOv2, and so do overlapping tiles of them, which show small objects
larger. A patch of the after photo with no similar patch nearby in the before photo is new; a
patch of the before photo with no similar patch nearby in the after photo is gone. The score
map is thresholded into a mask and boxes, each labelled appeared, disappeared or changed.

On testCYWS (28 before/after pairs of phone photos, 47 annotated changes, 15 pairs where
nothing changed), scored with Change Studio's own metrics:

| setting | changes found | pixel F1 | false-alarm pairs |
|---|---|---|---|
| whole photo only (board bf16) | 27 / 47 | 0.566 | 1 / 15 |
| whole photo + 0.6 tiles (the app's balanced preset) | 28 / 47 | **0.614** | 2 / 15 |
| + tiles, recall-leaning (threshold 0.35) | 30 / 47 | 0.603 | 3 / 15 |
| CYWS bf16, trained for change detection ([sima-cyws](https://huggingface.co/dotimothy/sima-cyws)) | 33 / 47 | 0.587 | 2 / 15 |

- **Caveat:** for both models the settings were tuned on these same pairs, so treat the numbers
  as optimistic for new scenes.
- The tile rows were scored with FP32 features. On the whole photo, the board's bf16 features
  score 0.566 against 0.569 for FP32, so the compiled model loses next to nothing here.
- The tile settings: tiles count only where the whole-photo map is at least 0.2, threshold 0.40,
  neighbourhood radius 0, border 1, smallest box 0.2% of the photo, blur 1.

## Input and output

- **Input:** one float32 448x448x3 tensor, HWC, RGB, pixel / 255. The ImageNet mean/std is
  inside the model.
- **Output:** float32 `(1, 384, 1, 1029)` after the final LayerNorm: 1029 = 1 class token +
  4 registers + 32 x 32 patch tokens, row by row. The runtime may hand it back channels last,
  as `(1029, 384)`.

## Using it

On a Modalix board, use [Neat Change Studio]({APP}). Its web demo lists the models in this
repository (`models.json`) and downloads one onto the board from the Models window
(Add Model). From a terminal:

```bash
webapp/run.sh --list-hub          # what is here and what is on the board
webapp/run.sh --fetch dinov2-448  # downloaded into models/ and checked
```

Without the app:

```bash
hf download {repo} dinov2-448/dinov2_vits14_reg_448_bf16_mpk.tar.gz --local-dir .
```

```python
import numpy as np, pyneat

opt = pyneat.ModelOptions()
opt.preprocess.kind = pyneat.InputKind.Tensor               # raw tensors in, no image preprocessing
model = pyneat.Model("dinov2-448/dinov2_vits14_reg_448_bf16_mpk.tar.gz", opt)

def tensor(hwc):
    t = pyneat.Tensor.from_numpy(np.ascontiguousarray(hwc, np.float32), copy=True)
    t.layout = pyneat.TensorLayout.HWC
    return t

image = np.zeros((448, 448, 3), np.float32)                 # your photo: RGB / 255, HWC
run = model.build([tensor(image)])
run.push_tensors([tensor(image)])
sample = run.pull(10000)
t = sample.tensors[0] if sample.tensors else sample.tensor
y = np.array(t.to_numpy(), np.float32, copy=True)
feat = y.reshape(384, -1).T if y.shape[-1] != 384 else y.reshape(-1, 384)   # (1029, 384)
cls, patches = feat[0], feat[5:].reshape(32, 32, 384)
run.close()
```

These files only run on the MLA. For PyTorch, use the [original weights]({DINO_UPSTREAM}).

## License and attribution

**Apache-2.0.** DINOv2, its code and pretrained weights, is by Meta Platforms, Inc.
({DINO_UPSTREAM}) and released under the Apache License 2.0.

These files are derived from those weights and are Apache-2.0 too:
- the network is rewritten to compile for the MLA, with the same function (within 5e-5);
- the ImageNet normalisation and the position embedding for 448x448 are folded in;
- it is compiled in bf16.

Nothing was retrained. The code that does this, and the change detection on top, is at
[dotimothy/neat-change-studio]({APP}).

```bibtex
@article{{oquab2023dinov2,
  title   = {{DINOv2: Learning Robust Visual Features without Supervision}},
  author  = {{Oquab, Maxime and Darcet, Timoth{{\\'e}}e and Moutakanni, Th{{\\'e}}o and others}},
  journal = {{Transactions on Machine Learning Research}},
  year    = {{2024}}
}}
@inproceedings{{darcet2024registers,
  title     = {{Vision Transformers Need Registers}},
  author    = {{Darcet, Timoth{{\\'e}}e and Oquab, Maxime and Mairal, Julien and Bojanowski, Piotr}},
  booktitle = {{International Conference on Learning Representations (ICLR)}},
  year      = {{2024}}
}}
```
"""


def c3po_card(repo: str, models: dict) -> str:
    def mb(e):
        return f"{sum(f['bytes'] for f in e['files']) / 1e6:.0f} MB"
    rows = "\n".join(
        f"| `{n}/{e['files'][0]['name']}` | {e['precision']} | {e['latency_ms']:g} ms | {e['accuracy']['changes_found']} | "
        f"{e['accuracy']['pixel_f1']:.3f} | {e['accuracy']['false_alarm_pairs']} | {mb(e)} |"
        for n, e in models.items())
    return f"""---
license: mit
library_name: sima-neat
pipeline_tag: image-segmentation
tags:
- sima.ai
- modalix
- mla
- edge
- change-detection
- c3po
---

# C-3PO change segmentation, compiled for the SiMa.ai MLSoC Modalix

[C-3PO]({C3PO_UPSTREAM}) ("How to Reduce Change Detection to Semantic Segmentation", Wang et
al., Pattern Recognition 2023) takes a **before** and an **after** photo and segments what
changed, pixel by pixel. A ResNet-18 shared by both photos feeds an MTF module that merges their
features as appeared, disappeared and exchanged; MSF fuses four scales and a DeepLabV3 head
produces the mask.

These are two of its released checkpoints compiled for the Modalix MLA with SiMa.ai's Model
Compiler. The whole network is **one MLA stage, with nothing on the CPU**, and one 256x256
before/after pair takes 46.7 ms on the MLA (bf16, `Model.benchmark`; 24.5 pairs/s at 12.8 W). A
plain pyneat loop that pushes a pair and pulls its result before the next took about 447 ms per
pair on the board, so expect more than the MLA time end to end.

| file | precision | MLA latency per pair | changes found | pixel F1 | false-alarm pairs | size |
|---|---|---|---|---|---|---|
{rows}

- **Latency:** `Model.benchmark` on a Modalix DevKit, plugins included.
- **Changes found, pixel F1, false alarms:** [Neat Change Studio]({APP})'s balanced preset
  (whole photo plus the 0.6 tiles where it saw a change; with all tiles, the accurate preset,
  PCD finds 30 / 47 and ChangeSim 22 / 47) on testCYWS: 28 before/after pairs of phone photos, with 47
  annotated changes and 15 pairs where nothing changed, scored with FP32 (bf16 agrees with
  FP32 on 99.98% of mask pixels). Neither checkpoint was trained on photos like these; CYWS
  ([sima-cyws](https://huggingface.co/dotimothy/sima-cyws)) finds 33 / 47 with F1 0.587.
- The checkpoints are binary (change / no change): boxes are not labelled appeared or
  disappeared.

## Input and output

- **Inputs:** `in0_before` and `in1_after`, each float32 256x256x3, HWC, RGB, pixel / 255. The
  ImageNet mean/std is inside the model.
- **Output:** float32 logits `(1, 2, 64, 64)`, class 0 = no change; the runtime may hand them
  back channels last, `(64, 64, 2)`. Change probability = 1 - softmax(logits)[0], upsampled
  bilinearly to the photo.

## Using them

On a Modalix board, use [Neat Change Studio]({APP}):

```bash
webapp/run.sh --fetch c3po-pcd    # or c3po-csb: downloaded into the models folder and checked
```

Without the app:

```python
import numpy as np, pyneat

opt = pyneat.ModelOptions()
opt.preprocess.kind = pyneat.InputKind.Tensor               # raw tensors in, no image preprocessing
model = pyneat.Model("c3po-pcd/c3po_pcd_256_bf16_mpk.tar.gz", opt)

def tensor(hwc):
    t = pyneat.Tensor.from_numpy(np.ascontiguousarray(hwc, np.float32), copy=True)
    t.layout = pyneat.TensorLayout.HWC
    return t

before = after = np.zeros((256, 256, 3), np.float32)        # your two photos: RGB / 255, HWC
run = model.build([tensor(before), tensor(after)])
run.push_tensors([tensor(before), tensor(after)])
sample = run.pull(10000)
t = sample.tensors[0] if sample.tensors else sample.tensor
y = np.array(t.to_numpy(), np.float32, copy=True).reshape(64, 64, 2)
change = 1 / (1 + np.exp(y[..., 0] - y[..., 1]))            # P(change), 64x64
run.close()
```

## License and attribution

**MIT.** C-3PO, its code and pretrained weights, is by Guo-Hua Wang et al. ({C3PO_UPSTREAM}),
under the MIT License. These files are derived from those weights and are MIT too: the
network is rewritten to compile for the MLA with the same function, and compiled in bf16.
Nothing was retrained.

```bibtex
@article{{wang2023c3po,
  title   = {{How to Reduce Change Detection to Semantic Segmentation}},
  author  = {{Wang, Guo-Hua and Gao, Bin-Bin and Wang, Chengjie}},
  journal = {{Pattern Recognition}},
  year    = {{2023}}
}}
```
"""


# One repository per family: what it holds, its model card and its note in the collection.
FAMILIES = {
    "cyws": {"repo": REPO, "models": MODELS, "card": CARD, "readme": cyws_card, "note": NOTE},
    "dinov2": {"repo": DINO_REPO, "models": DINO_MODELS, "card": DINO_CARD, "readme": dinov2_card, "note": DINO_NOTE},
    "c3po": {"repo": C3PO_REPO, "models": C3PO_MODELS, "card": C3PO_CARD, "readme": c3po_card, "note": C3PO_NOTE},
}


def document(family: str, names) -> dict:
    """models.json: version 2 adds `family` (top level and per entry) and `input`; older apps ignore both."""
    models = {n: entry(n, FAMILIES[family]["models"][n], family) for n in names}
    return {"app": "neat-change-studio", "version": 2, "family": family, "models": models}


def dry_run(family: str, repo: str, out: Path) -> int:
    """Write models.json and README.md for the builds found here, and touch no network."""
    fam = FAMILIES[family]
    names = [n for n, s in fam["models"].items() if (ROOT / s["build"]).is_file()]
    missing = [f"{n} ({fam['models'][n]['build']})" for n in fam["models"] if n not in names]
    doc = document(family, names)
    out.mkdir(parents=True, exist_ok=True)
    (out / "models.json").write_text(json.dumps(doc, indent=1) + "\n")
    (out / "README.md").write_text(fam["readme"](repo, doc["models"]))
    print(f"dry run, nothing uploaded: {out}/models.json and README.md for {repo}")
    print(f"listed: {', '.join(names) or 'nothing'}")
    if missing:
        print(f"no build for: {', '.join(missing)}")
    print(f"collection: {COLLECTION!r}: {COLLECTION_ABOUT}")
    print(f"collection note: {fam['note']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=list(FAMILIES), default="cyws")
    ap.add_argument("--repo", help="default: the family's repository")
    ap.add_argument("--only", nargs="+", help="upload only these models of the family")
    ap.add_argument("--list-only", action="store_true", help="write models.json and the card, upload no models")
    ap.add_argument("--no-collection", action="store_true")
    ap.add_argument("--dry-run", nargs="?", const="out/hub", metavar="DIR",
                    help="write models.json and README.md into DIR/<repo name> (default out/hub) instead of uploading")
    args = ap.parse_args()
    fam = FAMILIES[args.family]
    repo = args.repo or fam["repo"]
    unknown = sorted(set(args.only or []) - set(fam["models"]))
    if unknown:
        ap.error(f"Unknown model for {args.family}: {', '.join(unknown)} (choose from {', '.join(fam['models'])})")
    if args.dry_run is not None:
        return dry_run(args.family, repo, Path(args.dry_run) / repo.split("/")[-1])

    from huggingface_hub import HfApi                    # only for a real upload
    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True, private=False)
    names = args.only or list(fam["models"])
    if not args.list_only:
        for name in names:
            path = ROOT / fam["models"][name]["build"]
            print(f"uploading {name}: {path.name} ({path.stat().st_size / 1e6:.0f} MB)", flush=True)
            api.upload_file(path_or_fileobj=str(path), path_in_repo=f"{name}/{path.name}", repo_id=repo,
                            commit_message=f"Add {name}: {path.name}")
    # The list holds what is in the repository, so a model appears in the app once uploaded.
    present = set(api.list_repo_files(repo))
    doc = document(args.family, [n for n, s in fam["models"].items() if f"{n}/{Path(s['build']).name}" in present])
    api.upload_file(path_or_fileobj=json.dumps(doc, indent=1).encode(), path_in_repo="models.json",
                    repo_id=repo, commit_message="Update models.json")
    api.upload_file(path_or_fileobj=fam["readme"](repo, doc["models"]).encode(), path_in_repo="README.md",
                    repo_id=repo, commit_message="Update the model card")
    print(f"listed: {', '.join(doc['models']) or 'nothing'}")
    if not args.no_collection:
        owner = repo.split("/")[0]
        found = next((c for c in api.list_collections(owner=owner, limit=100) if c.title == COLLECTION), None)
        slug = found.slug if found else api.create_collection(COLLECTION, description=COLLECTION_ABOUT, private=False).slug
        coll = api.get_collection(slug)
        if found and coll.description != COLLECTION_ABOUT:
            api.update_collection_metadata(slug, description=COLLECTION_ABOUT)
        item = next((i for i in coll.items if i.item_id == repo), None)
        if item is None:
            api.add_collection_item(slug, repo, "model", note=fam["note"])
        elif item.note != fam["note"]:
            api.update_collection_item(slug, item.item_object_id, note=fam["note"])
        print(f"collection: https://huggingface.co/collections/{slug}")
    print(f"repository: https://huggingface.co/{repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
