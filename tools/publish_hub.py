#!/usr/bin/env python3
"""Publish the compiled CYWS models to Hugging Face, for the web demo's model catalog.

Each model is one MPK (`cyws_<name>_mpk.tar.gz`), uploaded into a folder of its name.
`models.json` at the top of the repository lists them with sizes and SHA-256 sums, and the
web app reads that list to offer the models for download (webapp/server.py, `Hub`): it
fetches a model onto the board, checks its sum, and can then load it on the MLA.
The repository also gets a model card (README.md), and is added to a collection.

    python3 tools/publish_hub.py                              # both models, card, list, collection
    python3 tools/publish_hub.py --only w8a16                 # one model (the list keeps the others)
    python3 tools/publish_hub.py --list-only                  # models.json and the card only

Needs `huggingface_hub` and a login with write access (`hf auth login`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
REPO = "dotimothy/sima-cyws"
COLLECTION = "SiMa.ai Change Detection Models"
COLLECTION_ABOUT = "Before/after change detection compiled for SiMa.ai Modalix: two photos in, boxes around what changed out, in one MLA pass."  # <= 150 chars
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


def entry(name: str, spec: dict) -> dict:
    path = ROOT / spec["build"]
    return {"title": spec["title"], "about": spec["about"], "path": name, "precision": spec["precision"],
            "latency_ms": spec["latency_ms"], "accuracy": spec["accuracy"], "card": CARD,
            "files": [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}]}


def card(repo: str, models: dict) -> str:
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

On a Modalix board, use [Neat Change Studio](https://github.com/dotimothy/neat-change-studio).
Its web demo lists the models in this repository (`models.json`) and downloads one onto the
board from Settings → Model. From a terminal:

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
[dotimothy/neat-change-studio](https://github.com/dotimothy/neat-change-studio).

```bibtex
@inproceedings{{sachdeva2023change,
  title     = {{The Change You Want to See}},
  author    = {{Sachdeva, Ragav and Zisserman, Andrew}},
  booktitle = {{Winter Conference on Applications of Computer Vision (WACV)}},
  year      = {{2023}}
}}
```
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--only", nargs="+", choices=list(MODELS))
    ap.add_argument("--list-only", action="store_true", help="write models.json and the card, upload no models")
    ap.add_argument("--no-collection", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=False)
    names = args.only or list(MODELS)
    if not args.list_only:
        for name in names:
            path = ROOT / MODELS[name]["build"]
            print(f"uploading {name}: {path.name} ({path.stat().st_size / 1e6:.0f} MB)", flush=True)
            api.upload_file(path_or_fileobj=str(path), path_in_repo=f"{name}/{path.name}", repo_id=args.repo,
                            commit_message=f"Add {name}: {path.name}")
    # The list holds what is in the repository, so a model appears in the app once uploaded.
    present = set(api.list_repo_files(args.repo))
    models = {n: entry(n, s) for n, s in MODELS.items() if f"{n}/{Path(s['build']).name}" in present}
    document = {"app": "neat-change-studio", "version": 1, "models": models}
    api.upload_file(path_or_fileobj=json.dumps(document, indent=1).encode(), path_in_repo="models.json",
                    repo_id=args.repo, commit_message="Update models.json")
    api.upload_file(path_or_fileobj=card(args.repo, models).encode(), path_in_repo="README.md",
                    repo_id=args.repo, commit_message="Update the model card")
    print(f"listed: {', '.join(models) or 'nothing'}")
    if not args.no_collection:
        owner = args.repo.split("/")[0]
        found = next((c for c in api.list_collections(owner=owner, limit=100) if c.title == COLLECTION), None)
        slug = found.slug if found else api.create_collection(COLLECTION, description=COLLECTION_ABOUT, private=False).slug
        coll = api.get_collection(slug)
        item = next((i for i in coll.items if i.item_id == args.repo), None)
        if item is None:
            api.add_collection_item(slug, args.repo, "model", note=NOTE)
        elif item.note != NOTE:
            api.update_collection_item(slug, item.item_object_id, note=NOTE)
        print(f"collection: https://huggingface.co/collections/{slug}")
    print(f"repository: https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
