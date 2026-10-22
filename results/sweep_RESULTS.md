# CYWS configuration tuning: results

Date 2026-10-22. Dataset `data/testCYWS`: 28 pairs. 15 of them have an empty GT mask; the
brief said 14, but `masks/20240918_173117` contains no pixel > 127. 47 GT components in total.
Model: FP32 PyTorch on the host RTX 5090 (`eval/torch_backend.py`), which matches ONNX FP32 to within
2.5e-4 in heat. Pipeline: the app as it is now, at `work_long=1920`, so the 12 MP pairs are decoded at
2104x1560 or 2000x1500. All metrics are computed in that working frame. (The short-side decode floor added later,
`max(64, work_long // 2)`, binds only above 2:1 aspect and leaves every testCYWS decode size unchanged.)

## TL;DR

| config (FP32 host, 28 pairs) | pooled F1 | pooled IoU | GT comps hit | fp pairs (of 15) | windows/pair | MLA ms/pair on board (est., 35 ms/win) | host CPU ms/pair* |
|---|---|---|---|---|---|---|---|
| old `levels 0`, thr 0.3 | 0.589 | 0.417 | 29/47 (0.617) | 1 | 1.07 | ~37 | 14.5 |
| old default (`levels 0,1`, tile auto, full, thr 0.3) | 0.589 | 0.418 | 31/47 (0.660) | 2 | 13.3 | ~465 | 23.0 |
| **`--preset fast`** (`levels 0`, thr 0.29) | 0.590 | 0.418 | 30/47 (0.638) | 2 | 1.07 | ~37 | 14.5 |
| **`--preset balanced` = new default** (`levels 0,1`, tile 0.45, gated at heat 0.25, thr 0.29) | **0.594** | **0.422** | **34/47 (0.723)** | 2 | **3.46** | ~120 (+ one gating round trip) | 17.9 |
| **`--preset accurate`** (`levels 0,1`, tile 0.45, every tile, thr 0.29) | 0.591 | 0.420 | 34/47 (0.723) | 2 | 13.3 | ~465 | 23.4 |
| balanced + `--register` (SIFT + verify) | 0.588 | 0.416 | 33/47 (0.702) | 2 | 5.36 | ~190 | 126 (+108 for SIFT and verify) |

\* Host CPU ms/pair is the app's own time with decode and inference wait taken out: `--workers 0`,
inference cache warm, cv2 using the host's 20 cores (`stages/8_final_serial`). Decoding a 12 MP pair at
`work_long=1920` adds about 85 ms on the host.

**What the default buys, in plain terms:** the new default finds 5 more GT components than levels 0 at
the old threshold (34 vs 29 of 47) and 3 more than the old two-level default. It runs **3.8x fewer
windows** than the old default (3.46 vs 13.3), so the MLA time drops from about 465 to about 120 ms per
pair. False alarms on the no-change pairs stay at 2.

**The pooled pixel F1 barely moves** (0.589 → 0.594), and that difference is noise; see the caveats. It
is capped by box-shaped masks against object-shaped GT and by a few very large components. No setting
in the grid moves it by more than ±0.01 without losing components.

The CLI ONNX FP32 run (`app/run_cyws_neat.py --preset balanced|fast --model-path
build/onnx/cyws_fused.onnx`, outputs in `build/cli_runs/onnx_*`) gives exactly the sweep's numbers:
balanced F1 0.594, 34/47, 2 fp pairs, 3.46 windows; fast F1 0.590, 30/47.

## Method

- `eval/sweep.py` with the cached GPU backend. Each pair is decoded once and every config shares the
  inference cache, so configs that differ only in post-processing do not recompute inference.
- **Registration is reproducible:** `registration.register()` now seeds cv2's RNG itself, so the
  registration runs give the same homography every time.
- **Metrics:**
  - **Pixel metrics:** pooled tp/fp/fn summed over all pairs.
  - **Component recall:** a GT component counts as hit when one box covers at least 50% of it, or the
    box and the component's bbox have IoU ≥ 0.3.
  - **fp pairs:** empty-GT pairs that get at least 1 box.
  - **Windows/pair:** this is the latency proxy, since the board needs ~35 ms of MLA per window, plus the
    crop.
- **Stages** (every directory has `sweep.csv`, `sweep_pairs.csv`, `sweep.md` and `sweep.log`):

| dir | grid | configs |
|---|---|---|
| `build/sweep/` (top level) | levels {0; 0,1; 0,1,2} × tile {auto, 0.35, 0.25} × tiles {full, gated} × thr {0.15, 0.2, 0.25, 0.3, 0.4, 0.5} × reg {off, on, on-diff}, full grid without pruning (234 configs, 228 s) | 234 |
| `stages/1_fine_thr_tile` | levels {0; 0,1} × tile {auto, 0.4, 0.3, 0.25, 0.2} × {full, gated} × thr 0.22–0.38 in steps of 0.02 | 99 |
| `stages/2_fine_thr_tile2` | tile {auto, 0.55, 0.45, 0.4, 0.35, 0.25} × thr 0.27–0.31 in steps of 0.01 | 65 |
| `stages/3_gate_coarse`, `stages/4_gate_x_tile` | gate_heat {0.05 … 0.4} × tile {auto, 0.45 … 0.25} × thr 0.26–0.32 (levels 0,1 and 0,1,2) | 588 |
| `stages/5_gate_x_tile_fine_reg` | gate_heat {0.26 … 0.34} × tile {0.375 … 0.5} × thr 0.27–0.31 × reg {off, on} | 610 |
| `stages/6_native_level2` | `work_long=0` (native), levels up to 0,1,2, thr down to 0.2: the small-object check | 150 |
| `stages/7_candidates` | the candidates at gate 0.25 | 120 |
| `stages/8_final_serial` | final candidates, single process, for host timing | 20 |

`gate_heat` was varied with `--set gate_heat=…`. A tile written as a fraction means that fraction of the
working frame's short side; auto is 0.5.

## What the sweep shows

1. **Threshold: optimum at 0.28–0.30, with a cliff above 0.31.** Pooled F1 moves in steps, because a few
   large components have peak scores right at 0.27 and 0.31 (e.g. on `20240920_184746` a 130k-pixel
   component scores 0.30–0.32). Example, levels 0,1 at auto tile:
   - F1 0.568 at thr 0.27, 0.587 at 0.28, 0.589 at 0.29, 0.590 at 0.30, 0.579 at 0.31, 0.542 at 0.32,
     then 0.47 at 0.4 and 0.38 at 0.5.
   - Below 0.27, false-alarm pairs climb (3–8) and F1 falls (0.41 at 0.15).

   **Choice: 0.29.** It sits in the middle of the 0.28–0.30 range. In the board's bf16 runs the scores of
   these borderline boxes moved by about ±0.02 against FP32; 0.30 sits 0.01 below the cliff, 0.29 has
   twice that margin. The old 0.30 gains 1 fp pair less but loses 1–2 components.
2. **Gated tiles are as accurate as full tiling, and cheaper.**
   - At every tile size and threshold, `gated` matches or beats `full` on F1, component recall and fp
     pairs, while running 2–4x fewer windows.
   - **Why:** the extra components tiling finds lie where level 0 already had some heat, and skipping the
     empty tiles also skips their false alarms.
   - `gate_heat` between 0.1 and 0.30 gives identical F1, recall and fp pairs at tile 0.45 and thr 0.29.
     Only the window count changes: 4.2 windows at 0.1, 3.46 at 0.25, 2.8 at 0.30.
   - Above 0.30 recall falls (34 → 31 components at 0.32).

   **Choice: 0.25.** It is safely inside the plateau, below the cliff and below thr.
3. **Tile 0.40–0.50 of the short side is a plateau; 0.30–0.375 is a dip.** Levels 0,1, gated, thr 0.29:

   | tile | 0.25 | 0.30 | 0.35 | 0.375 | 0.40 | 0.425 | **0.45** | 0.475 | 0.50 (auto) | 0.55 |
   |---|---|---|---|---|---|---|---|---|---|---|
   | F1 | 0.590 | 0.565 | 0.555 | 0.588 | 0.588 | 0.595 | **0.594** | 0.589 | 0.591 | 0.586 |
   | comps hit /47 | 34 | 34 | 33 | 34 | 33 | 33 | **34** | 33 | 32 | 32 |
   | fp pairs | 3 | 3 | 3 | 3 | 2 | 2 | **2** | 2 | 2 | 2 |

   - **Where the cells come from:** gate_heat 0.25 for tiles 0.25, 0.30, 0.35, 0.45 and auto; 0.26 for
     0.375–0.475 (stage 5); 0.1 for 0.55. At gate 0.25, tile 0.40 has 3 fp pairs instead of 2.
   - **Plateau around the choice:** over tile 0.40–0.50 × thr 0.28–0.30 × gate_heat 0.1–0.30 (90
     configs), F1 stays in 0.583–0.596, 31–34 components, 1–3 fp pairs, 2.6–5.6 windows.
   - **Why the dip:** at 0.30–0.375 the grid cuts some large objects into pieces that replace the level-0
     box.
   - **Choice: 0.45.** It is the middle of the plateau. It also has the best recall there, with fewer
     windows than auto, because smaller tiles gate more selectively.
   - **New in the app:** `AppConfig.tile` and `--tile` accept a float fraction, so the default scales
     with the frame: 702 px at 2104x1560, 486 px at 1920x1080.
4. **Level 2 and native resolution do not help on this data.**
   - **Level 2:** at `work_long=1920` and tile 0.45, the level-2 side (s1/2 = 351 px) is below the
     384 px minimum, so level 2 never runs. With tile auto it runs 7–43 windows and gains no F1 (0.589).
   - **Native resolution:** `work_long=0` gives F1 0.575–0.584 at best, and its metrics are computed in the
     native frame.
   - **The one thing that finds the 60 px components of `20240925_114502`:** native resolution, levels
     0,1,2, tile 0.25 and thr 0.2, which hits 4 of 7. That config scores F1 0.45, 12 fp pairs and 178
     windows per pair. It is a specialist mode, not a preset.
5. **Registration + verify: a small accuracy effect either way; its CPU cost keeps it out of the presets.**
   - **At levels 0 (the fast preset's structure) it helps slightly at thr 0.29:** F1 0.5906 vs 0.5896,
     the same 30/47 components, and 1 fp pair instead of 2, because it removes the `20240919_171935`
     false alarm (`stages/8_final_serial`, `L0-on-thr0.29` vs `L0-off-thr0.29`). At thr 0.30 it costs
     1 component (28 vs 29/47, F1 0.584 vs 0.589) with the same 1 fp pair.
   - **At levels 0,1 it does not help:** F1 is the same or lower (+0.001 to −0.016) and it finds 0–2
     components fewer (balanced: 33 vs 34/47, still 2 fp pairs); with tile auto, gated, thr 0.29 it adds
     a fp pair (`20240923_181709`). Gating on the diff map adds about 2 windows.
   - **Diff masks** (`on-diff`) are clearly worse: F1 about 0.42.
   - **Cost on the host:** register (SIFT 4000 at 1024 px, MAGSAC) is 97 ms plus a 6 ms warp at
     `work_long=1920`, 8 cv2 threads. Verify adds 6–12 ms.
   - **Cost on the board:** the board measured 804 ms of register at native resolution, where the host
     needs 113 + 34 ms. At the same ratio of about 5.5x, expect **~550–600 ms register plus ~100–250 ms
     verify per pair on the A65**, against ~120 ms of MLA for the whole balanced pipeline.
   - **Not in any preset, because of cost:** ~0.65–0.85 s of A65 time per pair buys at most one fewer
     false-alarm pair, against ~37 ms of MLA for the whole fast pipeline. For handheld or moved-camera
     captures, `--preset fast --register` is the option: on this data it keeps fast's recall and removes
     the parallax false alarm on `171935`, for that CPU time. See the failure analysis below.

## Chosen defaults and presets

Defaults are in `app/common.py`; `common.PRESETS` and `--preset` define the presets. Any flag given
explicitly overrides its preset.

| field | fast | **balanced (= AppConfig defaults)** | accurate |
|---|---|---|---|
| levels | 0 | 0,1 | 0,1 |
| tile | (0.45, unused) | 0.45 | 0.45 |
| tiles | (gated, unused) | gated | full |
| gate_heat | 0.25 | 0.25 | (unused) |
| thr | 0.29 | 0.29 | 0.29 |
| register | off | off | off |

- **fast:** 1 window per pair; the cheapest configuration that keeps 30/47 components.
- **balanced:** the best recall and the best F1 on the plateau, at about 3.5 windows.
- **accurate:** the same tile and threshold, but every level-1 tile runs. On this dataset it measures the
  same as balanced (0.591, 34/47, 2 fp) at 3.8x the windows. It is the safe choice when a change can
  leave no heat at all at level 0, which gating can never recover. Registration was left out of it
  because at levels 0,1 it measured worse here (1 component fewer, no fp pair removed).

## Per-pair breakdown of the default (balanced, FP32, ONNX CLI run)

| pair | size | windows | boxes | GT comps hit | pixel F1 | note |
|---|---|---|---|---|---|---|
| 20240916_175631 | 1920x1080 | 4 | 2 | 2/2 | 0.94 | |
| 20240916_180158 | 1920x1080 | 4 | 0 | no change | 1.00 | |
| 20240918_004758 | 4000x3000 | 6 | 4 | 3/3 | 0.73 | |
| 20240918_011930 | 4000x3000 | 1 | 0 | no change | 1.00 | |
| 20240918_150540 | 4208x3120 | 3 | 1 | 1/1 | 0.85 | |
| 20240918_150812 | 4208x3120 | 5 | 5 | 4/4 | 0.78 | |
| 20240918_154454 | 4208x3120 | 1 | 0 | no change | 1.00 | |
| 20240918_154538 | 4208x3120 | 4 | 3 | 1/1 | 0.52 | box mask over-covers (P 0.36) |
| 20240918_155618 | 4208x3120 | 1 | 0 | no change | 1.00 | |
| 20240918_155655 | 4208x3120 | 3 | 2 | 1/1 | 0.48 | box mask over-covers (P 0.33) |
| 20240918_173117 | 4208x3120 | 2 | 1 | no change (empty mask) | 0 | **fp**: a chair that turned (0.67, "changed"); arguably a missing label |
| 20240918_173205 | 4208x3120 | 5 | 1 | 1/1 | 0.70 | |
| 20240919_152613 | 4208x3120 | 1 | 0 | no change | 1.00 | |
| 20240919_152956 | 4208x3120 | 9 | 5 | 4/5 | 0.76 | |
| 20240919_170629 | 4208x3120 | 1 | 0 | no change | 1.00 | |
| 20240919_171750 | 4208x3120 | 8 | 15 | 10/10 | 0.76 | all 10 found; was 7–8 of 10 |
| 20240919_171935 | 4208x3120 | 4 | 4 | no change | 0 | **fp**: camera moved, parallax around the condiment caddy (scores 0.29–0.34) |
| 20240919_171958 | 4208x3120 | 6 | 1 | 0/3 | 0 | **miss**: viewpoint change, see below |
| 20240920_180702 | 4208x3120 | 1 | 0 | no change | 1.00 | |
| 20240920_181250 | 4208x3120 | 6 | 5 | 3/4 | 0.54 | |
| 20240920_184321 | 4208x3120 | 1 | 0 | no change | 1.00 | |
| 20240920_184746 | 4208x3120 | 9 | 7 | 4/5 | 0.62 | big component scores 0.30–0.32 (the thr cliff) |
| 20240923_180120 | 4208x3120 | 1 | 0 | no change | 1.00 | |
| 20240923_181709 | 4208x3120 | 7 | 0 | no change | 1.00 | had 2 boxes at thr ≤ 0.28 |
| 20240925_113632 | 4208x3120 | 1 | 0 | no change | 1.00 | |
| 20240925_114502 | 4208x3120 | 1 | 0 | 0/7 | 0 | **miss**: seven ~60 px screws, see below |
| 20240925_123536 | 4208x3120 | 1 | 0 | no change | 1.00 | |
| 20240925_123700 | 4208x3120 | 1 | 0 | no change | 1.00 | |

Pooled over all pairs: P 0.503, R 0.724, F1 0.594, IoU 0.422. Mean per-pair F1 is 0.738 and box
precision is 0.714.

### Why the failures fail (overlays inspected)

- **`20240919_171958`, 0/3 hit:**
  - **What happened:** before and after were shot from clearly different positions. The camera moved
    right and forward, so a chair, the window and the left chairs enter or leave the frame.
  - **The GT:** the three components (≈ 980x1230, 740x1230 and 540x1010 native px) are those regions,
    mostly at the frame border.
  - **The output:** CYWS sees no object-level change and puts one low-score "disappeared" box (0.31) on a
    chair back in the middle.
  - **What would fix it:** this is a viewpoint and border problem, not a threshold problem. A homography
    cannot align a deep 3D scene; the warp's invalid border would be the right signal for such
    components, if the labels mean "visible in only one image".
- **`20240919_171935`, no-change pair with 4 boxes:** the same scene, also with camera motion. The boxes
  sit on the condiment caddy and the chairs behind it, where parallax changes what is occluded, at
  scores 0.29–0.34. Raising thr removes them only by also losing large true components (see the
  threshold cliff). Registration removes them at levels 0 (`--preset fast --register`), but not at
  levels 0,1, where the level-1 tiles still fire on the parallax.
- **`20240925_114502`, 0/7 hit:**
  - **The change:** screws on three wall plates. Each is ~60 px native, ~30 px in the working frame.
  - **Why it is missed:** at level 0 a screw is about 4 model pixels, below `min_box_side=6`. Even a
    0.25 tile at native resolution makes it only ~20 px.
  - **What finds some:** only native resolution with level 2 and thr 0.2 finds 4 of 7, at an
    unacceptable false-alarm cost. Fixing this needs a small-object mode, e.g. tiles chosen around a
    region of interest, or a model trained for it.
- **`20240918_173117`:** the GT mask is empty, but the after image shows a chair rotated, with a
  specular highlight. The model says "changed", 0.67. It counts as a false alarm, though the label is
  debatable.
- **Low pixel precision on the large hits** (`154538`, `155655`, `184746`): box masks fill the whole
  rectangle, while the GT is a silhouette. Diff masks, the only alternative, scored worse here.

## Caveats and overfitting risk

- **28 pairs, 47 components, 15 no-change pairs: every number here is coarse.**
  - **One component is 2.1 points of recall,** and one fp pair is 1/15.
  - **Pooled pixel F1 rests on a few pairs:** leave-one-pair-out moves the default's F1 between 0.56 and
    0.65. A pair-bootstrap 95% interval is 0.43–0.71 for every config in the TL;DR table.
  - **The F1 gains are not significant:** the paired bootstrap difference against the old default is
    +0.004, 95% interval [−0.003, +0.015].
  - **What the evidence does support:** component recall (34 vs 29–31) and the 3.8x cut in windows,
    which is a mechanical effect of gating.
- **The defaults were picked on the same 28 pairs they are scored on; there is no held-out set.** To
  limit that, every choice was made on a plateau:
  - the middle of the 0.28–0.30 thr band and of the 0.40–0.50 tile band;
  - gate_heat inside the wide 0.1–0.30 flat region.

  Even so, the 0.45 vs 0.5 tile choice is worth 2 components, and that may not transfer. Re-check it on
  new data with `eval/sweep.py` (about 5 minutes for the full grid with a warm cache, 16 workers).
- **FP32 on the host, bf16 on the board:**
  - **Score shifts:** board scores differ by about ±0.02 for borderline boxes, and the earlier board run
    had box totals one off from the host. That is why thr is 0.29 rather than the cliff-edge 0.30.
  - **Not yet run on the board:** the new defaults. Run `--preset balanced` and `--preset fast` there to
    confirm.
- **Metrics in the working frame:** masks are compared at 2104x1560 or 2000x1500, so F1 is not strictly
  comparable to the native-frame board numbers (0.574). At native resolution the same configs score
  about 0.01 lower.
- **Latency estimates** use 35 ms per window from the bf16 benchmark. Gated mode needs one extra round
  trip, because level 0 has to finish before the gate decides. Load and preprocess times on the board
  are the speed track's host-to-board extrapolations, not measurements.
- **Disk:** `build/infer_cache` is now **33 GB** (float16 outputs for every distinct crop in all
  stages). It can be deleted at any time; it only speeds up re-sweeps.
