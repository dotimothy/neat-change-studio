# Neat Change Studio

Before/after change detection with CYWS on the SiMa.ai Modalix MLA, through NEAT.

Give it a **before** photo and an **after** photo of the same scene. It returns boxes around what changed (labelled *appeared*, *disappeared* or *changed*), a change mask and an overlay. The model is the original **"The Change You Want to See"** (Sachdeva & Zisserman, WACV 2023; [paper code](https://github.com/ragavsachdeva/The-Change-You-Want-to-See)): a Siamese ResNet-50 U-Net with co-attention and a CenterNet head. It runs on the Modalix MLA through NEAT (`pyneat`) as **one MLA ELF with no A65 stages**, at about 35 ms per 256×256 pair in bf16.

High-resolution photos are tiled. One global window runs first, then 256-input tiles are run only where the global view saw some change. This lets small objects be found at the scale the model was trained on.

It is a sibling of [Neat Decision Studio](https://github.com/dotimothy/neat-decision-studio). The compiled models are on Hugging Face in the collection [SiMa.ai Change Detection Models](https://huggingface.co/collections/dotimothy/simaai-change-detection-models-6ada4e701e447e9db528596b). Results and measurements are in [report.md](report.md). The project page is at **[timothydo.me/neat-change-studio](https://timothydo.me/neat-change-studio/)**.

![A test pair detected on the board: the boxes draw in, a detection is highlighted, then the before and after photos are compared with the swipe slider](docs/detect.gif)

## What you need

**To run it:**
- A Modalix DevKit with the NEAT SDK 0.5.0 (platform 2.1.3) and its `pyneat` venv. The scripts look for it in `~/pyneat`; set `PYNEAT_VENV` if it is elsewhere.
- ssh from the host to the board (passwordless is easiest; `deploy.sh` and `run_all.sh` use ssh and rsync).
- On the board, numpy, OpenCV (4.4 or later, for SIFT) and Pillow in the pyneat venv, and Flask for the web demo. `./setup.sh` checks them and `./setup.sh --install` installs what is missing ([requirements.txt](requirements.txt)).

**To build the model from scratch** (not needed if you download it):
- [uv](https://docs.astral.sh/uv/) and Python 3 on the host, with PyTorch installed. `run_all.sh` makes `.venv` with `--system-site-packages`, so it reuses that PyTorch; the other host packages are in [requirements-dev.txt](requirements-dev.txt).
- The SiMa NEAT SDK container with the Model Compiler. `compile/compile.sh` runs the compiler with `docker exec` in the container named by `CONTAINER` (default `ghcr.io-sima-neat-sdk-latest`). The repo must be under the host folder mounted as the container's `/workspace`, given by `WS_HOST` (default `~/SDK/NEAT/workspace`).
- `pyneat` comes from the NEAT SDK, and `afe` / `sima_utils` from the Model Compiler container. Neither is on PyPI.

## Get the model

The compiled models are on Hugging Face at [dotimothy/sima-cyws](https://huggingface.co/dotimothy/sima-cyws): `bf16` (the default, most accurate) and `w8a16`. Any of these puts them in `models/`:

```bash
webapp/run.sh --fetch bf16       # on the board; or w8a16, or all
webapp/run.sh --list-hub         # what Hugging Face has, and what is here
./setup.sh --fetch               # check the board environment, then fetch bf16 (--fetch=w8a16, --fetch=all)
```

The web demo can also download them: **Settings → Model**, then **Download**. Every download is checked against its SHA-256.

To compile the model yourself, see [Building it from scratch](#building-it-from-scratch).

## Quick start on the DevKit

From the host, copy the app, the web demo and the scripts to the board. `BOARD` is required; `REMOTE` defaults to `/media/nvme/cyws`. It refuses while another MLA app runs on the board, unless `FORCE=1`.

```bash
BOARD=sima@<devkit-ip> ./deploy.sh
# or also copy a model you compiled yourself:
BOARD=sima@<devkit-ip> ./deploy.sh build/bf16/cyws_bf16_mpk.tar.gz
```

Then, on the board:

```bash
cd /media/nvme/cyws
webapp/run.sh --fetch bf16       # unless you deployed an MPK
# one pair
./run.sh --mode static --before before.jpg --after after.jpg --out out/one
# a whole dataset, with metrics against the GT masks
./run.sh --mode dataset --dataset-dir data/testCYWS --out out/testcyws
```

`run.sh` activates `~/pyneat` (or `PYNEAT_VENV`) and defaults to `--model-path models/cyws_bf16_mpk.tar.gz`. Only one app may use the MLA at a time, so stop other demos first.

### Presets

| preset | what it runs | testCYWS (bf16 on the board) |
|---|---|---|
| `fast` | the global window only | 29 of 47 changes found, ~0.35 s per 12 MP pair (dataset mode) |
| `balanced` (default) | global window, then tiles of 0.45 × the short side where the global heat is ≥ 0.25 | 33 of 47, ~0.41 s per pair, 3.4 windows on average |
| `accurate` | global window plus every tile | finds what balanced finds, at ~4× the MLA time |

Any flag given explicitly overrides the preset. The ones you are most likely to want:

| flag | meaning |
|---|---|
| `--thr 0.29` | detection threshold on the heatmap |
| `--levels 0,1` / `--tiles gated\|full` / `--tile 0.45` | tiling levels, gating, tile side (pixels, or a fraction of the short side) |
| `--register` | align before to after on the CPU first (SIFT + MAGSAC homography); use for handheld captures. Turns on `--verify-change`, which drops boxes whose patches still match after the warp. Costs ~0.6–0.8 s of A65 time per pair. |
| `--mask-mode box\|diff` | filled boxes, or a pixel-difference mask inside each box (needs `--register`) |
| `--work-long 1920` | decode JPEGs at reduced resolution (long side ≥ 1920); `0` = native |
| `--backend neat\|onnx\|afe\|fake` | board MLA, host ONNX FP32, host compiler simulator, or a synthetic stand-in |
| `--bench` | per-stage timing table |

### Outputs

For every pair: `<id>_overlay.jpg`, `<id>_mask.png` and `<id>.json`. The JSON holds boxes in the working frame (`box`) and in original pixels (`box_orig`), labels, scores, windows and timings. Dataset mode adds `summary.csv` (one row per pair) and `metrics.json` (means plus pooled pixel IoU/F1).

## Dataset

The numbers in [report.md](report.md) come from testCYWS, the author's private set of 28 before/after phone-photo pairs. It is not published, so bring your own pairs in this layout:

```
data/testCYWS/
  images/<id>_ref.jpg    before
  images/<id>_cap.jpg    after
  masks/<id>.jpg         change mask (optional; needed for metrics)
```

The default location is `data/testCYWS` in the repo; set `CYWS_DATA` to use another folder (the scripts' `--dataset-dir` / `--data` flags also take a path). `deploy.sh --with-data` copies it to the board. If your pairs are on Google Drive, `data/download_testcyws.sh MANIFEST` fetches them file by file from a manifest you provide (see [data/testcyws_manifest.example.tsv](data/testcyws_manifest.example.tsv)).

## Web demo

```bash
# on the board (stop other MLA apps first)
/media/nvme/cyws/webapp/run.sh --daemon      # http://<board-ip>:8098/  (https:// on the same port for the webcam)
/media/nvme/cyws/webapp/run.sh --stop
```

Pick a test pair or upload two photos (on a phone you can take them with the camera). Choose a preset or adjust settings, and see the boxes, heatmap, tiles and timing. It also works on phones. Every screenshot here was taken on a Modalix DevKit, so the timings are the board's own.

**The start screen and a result.** A result shows the boxes, labelled *appeared*, *disappeared* or *changed*, over the after photo with the heat map. Next to it are the MLA time per window, the whole pipeline's time and, for test pairs, the score against the annotated mask.

<table>
<tr>
<td width="50%"><img src="docs/landing.png" alt="The start screen: the SiMa.ai logo, See What Changed, and four test pairs to try"></td>
<td width="50%"><img src="docs/result.png" alt="A result: four changed objects boxed on the after photo, with the timing and accuracy chips above and the test set beside it"></td>
</tr>
</table>

**Comparing.** Overlay, side by side, or a swipe slider between before and after. Layers for the heat map, the change mask, the ground truth and the tiles the MLA ran.

![Layers switched on and off on a test pair: heat map, ground-truth outline, change mask and the tiles, then side by side and the swipe slider](docs/compare.gif)

**Tuning.** Presets (Fast, Balanced, Accurate), the windows, the threshold and photo alignment. Each change re-runs straight away. The decoded photos are cached, so a re-run takes about half a second on the board.

![Switching presets and dragging the threshold slider: each change re-runs on the MLA and the boxes update](docs/tuning.gif)

**The whole test set.** Run every test pair with the current settings, scored against the annotated masks: changes found, false alarms, time per pair and F1, pair by pair.

![Run All Pairs: the 28 test pairs run one after another on the MLA, filling the table, and end at 33 of 47 changes found with 2 false alarms in 15 no-change pairs](docs/runall.gif)

**Webcam.** Capture **Before**, change something, then capture **After**. A self-timer (3, 5 or 10 s) counts down before each shot, and a faint ghost of Before helps line up After. **Watch Live** then compares every new frame with Before and keeps the result up to date. Here the camera is fed one of the test pairs.

![The webcam panel: a 3 s countdown captures Before, the scene changes, After is captured and detected, then Watch Live keeps detecting while a small live preview floats over the result](docs/webcam.gif)

**Models.** Settings → Model lists the compiled models on Hugging Face. A model that isn't on the board downloads with progress and a checksum check, and picking one switches the MLA over.

![Settings → Model: bf16 is loaded, w8a16 is picked, and the switch card steps through releasing the MLA, loading the model and warming up](docs/models.gif)

**On a phone.** The same page, with a tab bar.

<table>
<tr>
<td width="32%"><img src="docs/phone.gif" alt="On a phone: a test pair is tapped and detected, then the Results and Test Set tabs, and another pair"></td>
<td width="68%"><img src="docs/phone.png" alt="Two phone screenshots: a result with its detections, and the settings with the model catalog"></td>
</tr>
</table>

On the host it can run without a board:

```bash
.venv/bin/python webapp/server.py --backend onnx --data data/testCYWS
.venv/bin/python webapp/server.py --backend fake --data data/testCYWS     # no model needed
```

The `onnx` backend uses `build/onnx/cyws_fused.onnx`, which `model/fetch_upstream.sh` and `model/export_onnx.py` make (or run `./run_all.sh` up to its `export` step).

**Server flags** (`webapp/server.py`; `webapp/run.sh` passes them on):

| flag | meaning |
|---|---|
| `--port 8098` | port for both HTTP and HTTPS |
| `--model bf16\|w8a16` | model to load at start |
| `--models DIR` | where the MPKs are (default `models/`) |
| `--data DIR` | test pairs to offer (default `data/testCYWS`) |
| `--backend neat\|onnx\|fake` | board MLA, host ONNX FP32, or a synthetic stand-in |
| `--hub-repo REPO` | Hugging Face repository to list models from (default `dotimothy/sima-cyws`; `''` turns the catalog off) |
| `--fetch NAME…` | download these models into `--models` and exit (`all` for every one) |
| `--list-hub` | list the models on Hugging Face, and which are here, and exit |
| `--no-https` | plain HTTP only |

**Models from Hugging Face:**
- **Settings → Model** lists the models on [dotimothy/sima-cyws](https://huggingface.co/dotimothy/sima-cyws), as Neat GenAI Studio and Decision Studio do. A model that isn't on the board shows a **Download** button.
- A download fetches the model into `models/`, with progress, cancel and resume, and checks its SHA-256 before the model can be loaded.
- On a fresh board, the first model downloaded loads by itself.

The right-hand panel collapses with the round button on its edge, and the header's Settings button opens it at Settings.

**Webcam:** the webcam button captures Before, then After, from the browser's camera and detects. **Watch Live** compares every new frame with Before and updates the view (about 2 frames/s on the board with a 1080p webcam, most of it capture and upload). Browsers only allow the camera on HTTPS pages, so port 8098 speaks both HTTP and HTTPS (the server checks whether each connection starts with a TLS handshake). `webapp/run.sh` makes a self-signed certificate in `webapp/certs/` on first start (with `openssl`), so the browser warns about it once. `server.py` run directly uses `--cert` / `--key` (default `webapp/certs/cert.pem` and `key.pem`) if they exist, and serves HTTP only otherwise. The plain-HTTP page links to the HTTPS one when you open the webcam. `--no-https` turns HTTPS off.

## Building it from scratch

`./run_all.sh` reproduces everything on the host, in order: venv → dataset → upstream code + checkpoint → reference check → ONNX export → probe compiles → calibration → bf16/w8a16 compile → FP32 reference → board evaluation. `SKIP_BOARD=1` skips the board, and `STEPS="export compile"` runs only some steps. The board steps need `BOARD=sima@<devkit-ip>` and passwordless ssh. See [What you need](#what-you-need) for the host requirements.

**Compilation notes:**
- **bf16 setting:** bf16 needs `XLA_FLAGS=--xla_backend_extra_options=-vectorize-loops=false,-vectorize-slp=false`; `compile/compile.sh` sets it.
- **Single-ELF check:** `compile/cyws_compile.py` exits with an error unless the MPK has exactly one MLA ELF and no A65 stage.
- **Rewrite for the MLA:** the rewrite folds BatchNorm and writes co-attention as Einsum/Softmax with Split/Concat flattening. It expands the scSE spatial gate with a 1→C ones-conv, because two broadcast multiplies fail to compile. It splits the 6144→256 decoder conv into three input-channel chunks, because the single conv fails after "large tensor support". Each compiler failure and its fix is recorded in [report.md](report.md).
- **Calibration:** uses real **paired** before/after crops from the app's own preprocessing (`data/make_calib.py`).
- **Input order:** the MPK's input order is checked by `compile/mpk_io.py`: `in0_before`, then `in1_after`.
- **int8:** full int8 fails quantization (see the report). w8a16 compiles, but bf16 is more accurate on the board, so bf16 is the default.

**Tests:** `.venv/bin/python -m pytest tests -q` (host only; the web demo tests are skipped without Flask).

## Model input contract

There are two inputs, always pushed together and in this order:
- `in0_before` and `in1_after`, each float32 256×256×3 HWC, RGB, pixel / 255, with no mean/std normalisation.

There is one fused output, `(10, 256, 256)` float32, with channels `heat_b, heat_a, wh_b(w,h), wh_a(w,h), off_b(x,y), off_a(x,y)`. The heat channels are already passed through a sigmoid.

## Layout

| path | what |
|---|---|
| `app/` | The application: deployed to the board as `/media/nvme/cyws/app`. `run_cyws_neat.py` is the CLI; [app/API.md](app/API.md) documents the modules. |
| `webapp/` | Browser demo (Flask), served from the board on port 8098. |
| `model/` | Upstream fetch, the MLA-friendly rewrite (`cyws_mla.py`), the reference check and the ONNX export. |
| `compile/` | Model Compiler driver (bf16 / w8a16 / int8), probes, MPK input-order check. |
| `data/` | Dataset download from a manifest, dataset inspection, calibration tensors. |
| `eval/` | Raw-output comparison against FP32, GPU backend, inference cache, parameter sweep. |
| `bench/` | `bench_on_board.py`: streamed throughput and `Model.benchmark` on the DevKit. |
| `tools/` | `publish_hub.py`, for maintainers: uploads the MPKs to Hugging Face and regenerates `models.json` (the list the app reads) and the model card. Needs `huggingface_hub` and write access to the repository. |
| `results/` | Small result files behind the numbers in the report: the parameter sweep (`sweep_RESULTS.md`) and the board-vs-FP32 comparisons (`eval/`). |
| `tests/` | pytest suite (host, no board needed). |
| `run.sh`, `setup.sh`, `deploy.sh`, `run_all.sh` | board runner, board environment check and model fetch, deploy, full reproduction. |

Not in the repository (made by the scripts): `build/`, `third_party/`, `.venv/`, `models/`, `out/` and `data/testCYWS/`.

## License

Apache License 2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)).

CYWS is by Ragav Sachdeva and Andrew Zisserman (The Change You Want to See, WACV 2023, https://github.com/ragavsachdeva/The-Change-You-Want-to-See). The authors have confirmed to the maintainer (2026) that CYWS, its code and pretrained weights, is released under the Apache License 2.0. This supersedes the CC BY-NC default mentioned in upstream issue #11 (2024). The upstream repository does not include a LICENSE file yet.

- **Derived files:** the checkpoint, the ONNX files and the compiled MPKs are derived from the authors' pretrained weights and are also Apache-2.0. None of them is in this repository; the MPKs are on [Hugging Face](https://huggingface.co/dotimothy/sima-cyws).
- **This repository's code** is Apache-2.0.
- **Fonts:** Inter and JetBrains Mono, in the web demo, are under the SIL Open Font License 1.1.
- **Trademarks:** SiMa.ai, Modalix, Neat and the logos in `webapp/web/assets` are trademarks of SiMa.ai and are not covered by the Apache License.

## Citation

```bibtex
@inproceedings{sachdeva2023change,
  title={The Change You Want to See},
  author={Sachdeva, Ragav and Zisserman, Andrew},
  booktitle={Winter Conference on Applications of Computer Vision (WACV)},
  year={2023}
}
```
