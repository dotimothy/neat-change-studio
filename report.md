# CYWS on Modalix: results

Date 2026-10-22. Board: Modalix DevKit, NEAT 0.5.0, platform 2.1.3. Model: original CYWS (ResNet-50 Siamese U-Net + co-attention + CenterNet, `pretrained-resnet50-3x-coam-scSE-affine`). Dataset: testCYWS, 28 before/after pairs of phone photos (4208×3120, 4000×3000 and 1920×1080). 15 pairs have an empty GT mask (no change), and there are 47 GT change components in total.

## Summary

- **One MLA ELF, 0 A65 stages:** the whole network compiles in bf16 to a single MLA ELF. Apart from that ELF the MPK has 7 EV74 plugins, and no stage runs on the A65.
- **35.2 ms per pair:** `Model.benchmark` gives 35.2 ms (36 fps) at 13.2 W. Streamed through the app with 2 pairs in flight, it is 28 ms per pair.
- **Board output matches FP32:** board bf16 vs ONNX FP32 on 97 inputs gives box agreement 1.0 (148/148 boxes), heat cosine ≥ 0.994 and AP50 0.928. The before/after swap test passes.
- **Board app on testCYWS (balanced preset, the default):** finds 33 of 47 changes, with pooled pixel F1 0.587. Two of the 15 no-change pairs get a false box. Over the whole dataset this takes 0.41 s per 12 MP pair. Decoding the next pair and writing outputs overlap the MLA work. Without that overlap a pair takes 0.45 s on average (0.6 s for one with many tiles), of which about 0.18 s is decoding.
- **Board vs host:** the board stays within 0.01 F1 and one component of FP32 on the host (34/47, F1 0.594).
- **int8 and w8a16:** full int8 fails quantization. w8a16 compiles, but on the board it is less accurate than bf16 and barely faster per pair, so bf16 stays the shipped model.

## 1. Model port

`model/cyws_mla.py` is a self-contained rewrite with no smp or mmdet dependency. It loads the upstream checkpoint strictly: 0 missing and 0 unexpected keys. (The upstream loader uses `strict=False`.)

Checks against the upstream code (`model/verify_reference.py`):

| check | result |
|---|---|
| rewrite vs upstream, float64 | max abs diff 1.2e-12 |
| rewrite vs upstream, float32 | heat ≤ 2.3e-6 |
| BN-folded vs unfolded, ORT vs torch | passes (identical peaks, box IoU > 0.999) |
| synthetic pasted objects | 4/4 detected |
| co-attention logit range (no 1/√d scaling) | up to 135.8 |
| bf16 autocast preview | box agreement 1.0, heat cosine 0.984 |

The rewrite made these changes for the compiler. Each one came from a probe compile that failed:

| pattern | problem | fix |
|---|---|---|
| co-attention probes | disconnected outputs compiled to 2 MLA segments | probes concatenate their outputs |
| scSE gate (`x·cSE + x·sSE`, two broadcast multiplies) | "Compile model failed" | expand the 1-channel spatial gate with a fixed 1→C conv of ones |
| first decoder conv, 6144→256 3×3 | fails after "large tensor support" | split it into 3 input-channel chunks and add the results |
| co-attention | — | `Split/Concat` flatten → `Einsum nchw,nchq->nqhw` → `Softmax(axis=1)` → `Einsum nchw,nqhc->nqhw` → unflatten |
| CenterNet head (3 branches × 2 sides) | — | fused into one 3×3 conv 64→192 and one block-diagonal 1×1 conv 192→5 per side |
| outputs | — | one fused `out (1,10,256,256)`, so outputs cannot be swapped |

The ONNX graph has 49.47M parameters, about 75 GMAC per pair, and uses only allowlisted ops.

## 2. Compilation

Compiled with the SiMa Model Compiler in `ghcr.io-sima-neat-sdk-latest`. Calibration uses 98 real before/after crops from the app's own preprocessing (globals and tiles).

| precision | result | MPK | notes |
|---|---|---|---|
| **bf16** | **1 MLA ELF, 0 A65** | 114 MB | needs the XLA no-vectorize flag |
| w8a16 (min-max) | 1 MLA ELF, 0 A65 | 68 MB | |
| int8 (min-max) | quantization failed | — | Concat input/output scale ratio too small to represent |
| int8 (MSE) | quantization failed | — | same Concat error; the head conv also saturates (weights set to zero) |

The int8 failure comes from the U-Net skip concatenations. Their inputs differ in range by over 200× (for example, input scale 1.39 vs output scale 337), and the head saturates. Fixing it would need per-branch rescaling in the graph, and bf16 already meets the latency need, so it was not pursued.

The MPK's input order was checked (`compile/mpk_io.py`): placeholder 0 is `in0_before` and placeholder 1 is `in1_after`.

## 3. Accuracy of the compiled model (raw outputs)

These runs use 97 inputs (28 global views, 64 tiles and 3 swap pairs), compared against ONNX FP32 (`eval/compare.py`). Heat cosine is computed only on inputs whose FP32 heat peak is at least half the threshold.

| | bf16 on the board | w8a16 on the board |
|---|---|---|
| heat cosine, min / mean | **0.994** / 0.999 | 0.826 / 0.974 |
| heat max abs error | 0.058 | 0.223 |
| wh/offset relative error, median / p95 | 0.003 / 0.014 | 0.014 / 0.062 |
| box agreement (148 FP32 boxes) | **1.000** | 0.966 |
| AP50 with FP32 boxes as GT | 0.928 | 0.956 |
| swap test (output nearer to its own order, 3 pairs) | pass (×17–22 margin) | pass |

Board bf16 vs the host compiler simulator: cosine ≥ 0.9998.

## 4. Latency

**Model only, bf16:**
- `Model.benchmark` (200 samples, plugin latency included): **35.2 ms per pair, 36 fps, 13.2 W**.
- Streamed through the app backend with 2 in flight: **28 ms per pair**.
- w8a16: 27.2 ms (`Model.benchmark`).

`bench/bench_on_board.py` repeats both measurements.

**Whole app per pair** (board, testCYWS, mean of 28 pairs, `--work-long 1920`, bf16):

| preset | windows/pair | dataset wall time per pair | load (decode + GT mask) | stream (crops + MLA + decode) |
|---|---|---|---|---|
| fast | 1.07 | **354 ms** | 181 ms | 170 ms |
| **balanced** | 3.39 | **409 ms** | 181 ms | 260 ms |

**Web demo, one 12 MP test pair** (board, balanced, the whole HTTP request): **~0.9 s** on the first run, **~0.55 s** for a re-run with new settings.

**Optimisation notes.** Earlier versions were slower, mostly because of CPU work on the A65. With every tile run at native resolution the app needed about 1.1–2.4 s per 12 MP pair. Before the decoding changes below, the dataset wall time was 590–643 ms (fast) and 677–744 ms (balanced), of which about 490 ms was load, and the web demo took ~1.5 s (~1.2 s for a re-run). The changes below brought these to the numbers above. They did not change accuracy: balanced 0.588 F1 and 33/47, fast 0.586 and 29/47, before and after.
- **Decoding at reduced resolution:** JPEG draft decoding (long side ≥ 1920) cut decode about 3×. A 12 MP phone JPEG still takes 90–120 ms on one A65 core, because entropy (Huffman) decoding is serial and does not shrink with the output size.
- **Parallel decoding:** the before photo, the after photo and the GT mask decode at the same time on separate cores. The before photo's size and the mask's frame are predicted from the JPEG headers, so nothing waits for the after photo.
- **GT mask clean-up at working resolution:** thresholding, opening and labelling run at working resolution instead of native 13 MP: 342 → ~145 ms, overlapped with the photos. Scores on the host are unchanged (balanced F1 0.5935 → 0.5946, same 34/47 and 2 false-alarm pairs).
- **Dataset mode overlap:** the next pair is decoded, and the previous pair's overlay, mask and JSON are written, while the current pair runs on the MLA. The overlay is blended with OpenCV uint8 operations instead of float32 numpy (a float32 blend took ~250 ms at 3 MP).
- **Web demo:**
  - Decoded pairs are kept (6 most recent), so a re-run with a new threshold or preset skips the decode.
  - The photo and ground-truth images are encoded once per pair, in parallel.
  - The heat layer is sent at its native 1/4 resolution (the browser upscales it), and the mask tint at half resolution. PNG encoding of a 1600 px RGBA layer alone was ~115–360 ms on the A65.
  - Phone photos are shrunk to a 2048 px long side in the browser before upload, which makes both the upload and the board's decode several times faster.
- **Not usable: the hardware decoder.** Modalix has a hardware JPEG decoder (`neatdecoder dec-type=jpeg`), but it is limited to 3840×2160, so it cannot take 12 MP photos. It is also built for video streams and produced no output for a single still image.

## 5. Change detection on testCYWS

**Metrics:**
- **Pixel metrics:** pooled over all pairs, filled boxes vs the GT mask, in the working frame.
- **Component found:** a box covers ≥ 50% of the component, or IoU ≥ 0.3 with its bbox.
- **False-alarm pairs:** no-change pairs that get at least one box.

| configuration | backend | pooled F1 | pooled IoU | components found | false-alarm pairs (of 15) | windows/pair |
|---|---|---|---|---|---|---|
| first board run (native res, every tile, thr 0.3) | board bf16 | 0.574 | — | — | — | 13 |
| global only, thr 0.3 | host FP32 | 0.589 | 0.417 | 29/47 | 1 | 1.07 |
| `fast` | host FP32 | 0.590 | 0.418 | 30/47 | 2 | 1.07 |
| `fast` | **board bf16** | 0.585 | 0.413 | 29/47 | 2 | 1.07 |
| `balanced` | host FP32 | 0.594 | 0.422 | 34/47 | 2 | 3.46 |
| `balanced` | **board bf16** | **0.587** | 0.415 | **33/47** | 2 | 3.39 |
| `balanced` | board w8a16 | 0.581 | 0.410 | 31/47 | 3 | 3.32 |
| `accurate` | host FP32 | 0.591 | 0.420 | 34/47 | 2 | 13.3 |
| `balanced` + `--register` | host FP32 | 0.588 | 0.416 | 33/47 | 2 | 5.36 |

The settings were tuned on these same 28 pairs (`eval/sweep.py`, about 1,900 configurations; details in [results/sweep_RESULTS.md](results/sweep_RESULTS.md)). With so few pairs, differences of about 0.01 F1 or one component are noise, so these numbers are optimistic for new scenes. The choices sit in the middle of flat regions rather than at sharp peaks:
- threshold 0.29: F1 drops off a cliff above 0.31;
- tile 0.45 × short side: the plateau runs from 0.40 to 0.50;
- gate heat 0.25: the plateau runs from 0.10 to 0.30.

Pixel F1 is capped near 0.6 by the output format: CYWS produces boxes, while the GT masks follow object outlines. The number of components found is the more useful measure.

**Where it fails:**
- `20240925_114502`: the changes are about 60 px objects in a 12 MP photo, and none is found. Only native resolution with three tiling levels at thr 0.2 finds 4 of 7, and that setting gives 12 false-alarm pairs and 178 windows per pair.
- `20240919_171935`: a false alarm caused by camera parallax. `--register` removes it with the `fast` preset.
- `20240918_173117`: the model reports one change, but the GT mask has no pixel above 127, so the pair counts as no-change. The mask may be too faint rather than the scene unchanged.
- `20240919_171958`: F1 0. Its one box does not overlap the GT change.

**Registration** (SIFT + MAGSAC, then a ZNCC check that drops boxes whose patches still match after the warp) is in no preset. It costs about 0.6–0.85 s of A65 time per pair and buys at most one fewer false alarm on this data. Use it for handheld captures.

## 6. Reproducing

The dataset is testCYWS in `data/testCYWS` (or `CYWS_DATA`). It is private, so with your own pairs the numbers will differ.
- **Whole pipeline:** `BOARD=sima@<devkit-ip> ./run_all.sh` (see the README; `SKIP_BOARD=1` for the host part only).
- **Board dataset runs** (on the board, in `/media/nvme/cyws`): `./run.sh --mode dataset --dataset-dir data/testCYWS --preset balanced --out out/tuned_balanced`, and the same with `--preset fast --out out/tuned_fast`. Each output folder holds `summary.csv` and `metrics.json`.
- **Raw comparison:** on the host, `.venv/bin/python eval/run_model.py --backend onnx --model build/onnx/cyws_fused.onnx --out build/eval/fp32`; on the board, `eval/run_model.py --backend neat --model models/cyws_bf16_mpk.tar.gz --inputs eval/eval_inputs.npz --out eval/board_bf16` (`run_all.sh`'s `board` step does this and copies the result back); then `.venv/bin/python eval/compare.py build/eval/fp32 build/eval/board_bf16 --out build/eval/compare_board_bf16_vs_fp32.json`. The results behind section 3 are in [results/eval/](results/eval/).
- **Parameter sweep:** `.venv/bin/python eval/sweep.py` (host, FP32; see `--help`). The summary is in [results/sweep_RESULTS.md](results/sweep_RESULTS.md).
