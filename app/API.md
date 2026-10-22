# app/ module reference

Types come from `app/common.py`. Modules import each other as plain modules
(`from common import Window`), because app/ is deployed as a flat directory and
run as `python3 run_cyws_neat.py`. Tests add app/ to `sys.path` via
`tests/conftest.py`. Only numpy, opencv-python (cv2), Pillow and the standard
library are guaranteed on the DevKit (Python 3.11, numpy 2.4, cv2 5.0 with
SIFT, PIL 9.4, scipy 1.10). onnxruntime exists only on the host, pyneat only on
the board, so import them lazily inside the backend that needs them.

## common.py (config)
- `AppConfig` - every tunable; its defaults are the tuned "balanced" preset
  (results/sweep_RESULTS.md): `levels=(0, 1)`, `tile=0.45` (fraction of the
  working frame's short side), `tiles="gated"`, `gate_heat=0.25`, `thr=0.29`,
  `register=False`, `work_long=1920`.
- `PRESETS: dict[str, dict]` - AppConfig overrides per preset: `fast` (levels
  0 only), `balanced` (= the defaults), `accurate` (levels 0,1 with every
  level-1 tile, `tiles="full"`); all use tile 0.45, gate_heat 0.25, thr 0.29,
  no registration.
- `preset_config(name="balanced", **overrides) -> AppConfig`; unknown name ->
  CywsError.

## dataset_loader.py
- `discover_pairs(root) -> tuple[list[PairItem], list[str]]`
  Layout: `root/images/<id>_ref.<ext>` (before), `root/images/<id>_cap.<ext>`
  (after), `root/masks/<id>.<ext>` (optional GT). Also accepts the files directly
  in `root`. ext in jpg/jpeg/png, case-insensitive. Returns pairs sorted by id
  plus human-readable warnings for orphans (ref without cap, mask without pair,
  etc.). Raises `CywsError` if root is missing or no complete pair exists.
- `static_pair(before, after, mask=None, pair_id=None) -> PairItem`
  Validates that the files exist and are files; pair_id defaults to the after
  file's stem.

## ingestion.py
- `short_floor_for(work_long) -> int` - `max(64, work_long // 2)`: the smallest
  short side a reduction may leave (binds only above 2:1 aspect).
- `reduction_for(long_side, work_long, max_factor=0, *, short_side=None) -> int` -
  largest power of two s with `long_side // s >= work_long` and, when short_side
  is given, `short_side // s >= short_floor_for(work_long)` (1 when work_long <= 0);
  max_factor caps it (8 for JPEG draft decoding).
- `read_image_rgb(path, work_long=0) -> tuple[np.ndarray, dict]` - PIL open,
  `ImageOps.exif_transpose`, RGB uint8 (H, W, 3). work_long > 0 decodes at
  reduced resolution keeping the long side >= work_long and the short side >=
  `short_floor_for(work_long)` (so a 7680x1080 panorama stays native instead of
  dropping to 1920x270 and losing level 1): JPEG/MPO via PIL draft
  mode (`Image.draft("RGB", (w // s, h // s))` before `load()`, DCT-domain 1/2,
  1/4, 1/8, result size ceil(w / s)); other formats (and a JPEG still >=
  2*work_long after 1/8) decode fully and are INTER_AREA-downscaled by a power
  of two only if both bounds still hold at half size. EXIF transpose is applied
  after draft. meta: `raw_size` (w, h stored, before transpose/reduction),
  `full_size` (w, h transposed, native), `size` (w, h returned), `scale`
  (full / returned, x and y), `reduction` (total factor), `draft` (decoder
  factor), `exif_orientation` (1..8), `mode`, `path`. Raises CywsError on
  unreadable / too small (< 64 px natively or after reduction) / work_long < 0.
- `read_mask(path, target_hw, raw_after_wh, after_orientation, out_hw=None) -> np.ndarray` (bool)
  target_hw is the after image's *native* transposed size. Grayscale; if it
  matches the raw (untransposed) after size, apply the same EXIF transpose; if
  same aspect (2%) but other size, nearest resize; else CywsError. Threshold
  > 127, 3x3 opening, drop components < 0.01% of image - all at native size;
  then, if out_hw differs, INTER_NEAREST resize into out_hw (the working frame).
- `load_pair(item, *, allow_squash=False, register=False, work_long=0) -> Pair`
  after defines the frame, decoded with `read_image_rgb(after, work_long)`; the
  canonical frame is that working-resolution frame. before is decoded with
  work_long = the after's working long side (so it is >= the frame, never
  needlessly upscaled). before: aspect within 2% -> resize to after size
  (INTER_AREA when shrinking); otherwise register -> keep its decoded size
  (meta["before_native"]=True); allow_squash -> resize with a warning in meta;
  else CywsError. GT mask: `read_mask(..., target_hw=native, out_hw=(H, W))`.
  meta: both images' read meta, `work_long`, `orig_size` (w, h native
  transposed after), `work_size` (w, h canonical), `scale` (sx, sy) = original
  after pixels per canonical pixel, `mask` (path, positive_frac, read_ms).
  The default work_long=0 keeps native resolution (calibration scripts).
- `resize_to(img, w, h) -> np.ndarray` - INTER_AREA if shrinking, else INTER_LINEAR.
- `preprocess(img_u8, win, interp="area") -> np.ndarray` - crop `win.slices()`,
  resize to (256, 256) (area: INTER_AREA when shrinking, INTER_LINEAR when
  growing; bilinear: always INTER_LINEAR), float32 / 255, C-contiguous (256,256,3).
- `crop_pair(before, after, win, interp="area") -> tuple[np.ndarray, np.ndarray]`
  asserts identical shapes, uses the *same* slices for both, returns two
  preprocessed tensors.

## tiling.py
- `axis_positions(L, s, overlap) -> list[int]` - n = 1 if L <= s else ceil((L-s)/(s*(1-overlap)))+1;
  positions round(i*(L-s)/(n-1)); a single window is centred.
- `auto_tile_size(H, W, median_component_long_side=None) -> int` - if a hint is
  given: s = 4*hint; else min(H,W)//2; clamp to [256, min(H,W)].
- `resolve_tile(tile, H, W) -> int | "auto"` - "auto" stays "auto"; an int is
  pixels (>= 1); a float in (0, 1] is a fraction of min(H, W):
  `max(64, round(tile * min(H, W)))`; anything else -> CywsError.
- `plan_windows(H, W, *, tile="auto", overlap=0.25, levels=(0,1), max_aspect_squash=1.5,
  median_component_long_side=None) -> list[Window]` (tile goes through
  `resolve_tile` first, so 0.45 on a 2104x1560 frame is 702 px)
  Level 0: aspect <= 1.5 -> one full-image window; else ceil(long/short) square
  windows of side `short` along the long axis (axis_positions with the overlap
  that results). Level 1: square side s1 = tile or auto, clamped to
  [256, min(H,W)], skipped if s1 >= 0.8*min(H,W). Level 2 (only if requested):
  s1//2, only if >= 384. If min(H,W) < 256, level 0 only. min(H,W) < 64 ->
  CywsError. Ids are sequential, level 0 first. img_edges computed per window.
- `window_to_canonical(boxes, win) -> np.ndarray` - boxes (N, >=4) in 256 coords
  -> canonical: X = x0 + x*w/256, Y = y0 + y*h/256 (other columns copied).
- `canonical_to_window(boxes, win)` - inverse.
- `valid_fraction(win, valid_mask) -> float`.
- `class HeatCanvas(H, W, scale=4)`: `.add(win, heat256)` writes per-pixel max of
  the heat resized into the window footprint at 1/scale (block max when
  shrinking, so narrow peaks survive); `.array` float32
  (ceil(H/scale), ceil(W/scale)); `.max_in(win) -> float`;
  `.footprint(win) -> (rows, cols)` canvas slices a window covers.
- `gate_windows(windows, canvas, diff_energy=None, gate_heat=0.1, diff_percentile=75.0) -> list[Window]`
  keeps windows whose canvas max >= gate_heat or whose mean diff energy is above
  the given percentile over those windows (diff_energy is an (H/4, W/4) map, same scale as the canvas).

## registration.py
- `register(before, after, method="sift", work_long=1024) -> RegResult`
  Seeds cv2's RNG (`cv2.setRNGSeed(RNG_SEED)`, 12345, per thread) first, so
  the same pair always gives the same H (USAC/RANSAC sampling and FLANN draw
  from it). Downscale both to long side work_long; SIFT(4000) (ORB(5000) if method="orb"
  or SIFT is unavailable); ratio test 0.75; `cv2.findHomography(USAC_MAGSAC, 3.0)`
  (fallback cv2.RANSAC if USAC is missing). Accept iff inliers >= 30,
  inlier_ratio >= 0.25, det of upper-left 2x2 in [0.5, 2], |h31|,|h32| < 2e-3
  at work scale, warped corners convex and not flipped. Else try
  `estimateAffinePartial2D` with the same inlier rules, else identity (ok=False).
  Lift to full res: H_full = S_after^-1 @ H_work @ S_before. valid_mask is the
  warped all-ones mask (bool, after-frame shape), eroded by 2 px.
  For identity with a different-sized before, H is the plain scale matrix.
- `warp_before(before, reg, after) -> np.ndarray` - warpPerspective into after's
  shape (INTER_LINEAR), then pixels outside reg.valid_mask are copied from
  `after`, so the unwarped border reads as "no change".
- `photometric_match(src, ref, valid_mask) -> np.ndarray` - per-channel histogram
  matching of src to ref over the valid region (uint8 LUTs).
- `abs_diff_gray(a, b) -> np.ndarray` - float32 |gray(a)-gray(b)|.
- `median_abs_diff(after, before_aligned, valid_mask=None) -> float` - median of
  abs_diff_gray over valid pixels (histogram-based, interpolated inside its
  bin so it is a float, not an integer gray level), the value to pass as
  verify_detection's global_median_diff.
- `diff_energy_map(after, before_aligned, valid_mask=None, scale=4) -> np.ndarray` - mean
  abs gray diff pooled to (ceil(H/scale), ceil(W/scale)), zero outside valid.
- `verify_detection(det, after, before_aligned, *, zncc_thr=0.85, diff_ratio=1.5,
  global_median_diff, shift_frac=0.03, min_shift=4) -> tuple[bool, dict]`
  Max ZNCC between the after patch and the before_aligned patch over shifts of
  +-max(min_shift, shift_frac*box size) (boxes over 256 px: exhaustive search
  at reduced scale, then refined coarse to fine to an exact full-resolution
  shift); returns keep=False iff zncc_max >
  zncc_thr and mean |diff| in box *at the best shift* < diff_ratio *
  max(global_median_diff, 1). Boxes smaller than 8 px on a side are always
  kept. Callers pass global_median_diff = median of `abs_diff_gray(after,
  before_aligned)` over the valid mask.

## post_process.py
- `decode_centernet(out, branch, thr=0.3, topk=100) -> np.ndarray` - out is the
  canonical (10,256,256). 3x3 local maxima of heat (equal to max-filter) with
  heat >= thr, top-k by score, cx = x + off_x, cy = y + off_y,
  box = (cx - w/2, cy - h/2, cx + w/2, cy + h/2). Returns (N, 5) float32
  [x1, y1, x2, y2, score] in 256 coords, sorted by score desc.
- `filter_window(dets256, win, branch, cfg) -> list[Detection]` - drop boxes with
  short side < cfg.min_box_side (256 units); at level >= 1, drop boxes within
  cfg.edge_margin of an interior window edge; clip to [0,256]; map to canonical;
  set window_id, level, branch.
- `drop_invalid(dets, valid_mask, max_invalid_frac=0.5) -> list[Detection]`.
- `iou_matrix(a, b)` -> (N, M) IoU of every box in `a` (N, 4) against every box in
  `b` (M, 4); boxes are (x1, y1, x2, y2), degenerate boxes give 0.
  `containment_matrix(a, b)` -> (N, M) with `out[i, j] = area(a[i] ∩ b[j]) / area(b[j])`,
  the share of `b[j]` inside `a[i]` (near 1 means a[i] contains b[j]);
  degenerate `b` boxes give 0. Not symmetric.
- `merge(dets, windows, cfg) -> list[Detection]` - one branch at a time:
  (0) seam rule (only when level 0 has several squares, aspect > 1.5): a
  level-0 box within cfg.edge_margin of an interior side of its square is
  dropped if an uncut level-0 box of another square contains it
  (> cfg.containment), else chained with facing cut boxes of neighbouring
  squares that overlap it > 0.5 along the seam; each chain becomes one union box;
  (1) huge-box rule: a level>=1 box covering > cfg.huge_box_frac of its window
  survives only if a level-0 box overlaps it with IoU > 0.3;
  (2) split rule: drop a level-0 box that contains (> 0.7 containment) >= 2
  level>=1 boxes whose union covers < 0.6 of it;
  (3) greedy by score: suppress j if IoU > cfg.nms_iou or inter/area_j > cfg.containment
  (cfg.merge == "wbf": score-weighted box average of each IoU>0.55 cluster instead);
  (4) keep at most cfg.max_boxes.
- `fuse_branches(after_dets, before_dets, iou=0.5) -> list[Detection]` - pairs
  with IoU > iou -> one "changed" detection (after box, max score); unmatched
  after -> "appeared"; unmatched before -> "disappeared".
- `make_mask(dets, hw, mode="box", after=None, before_aligned=None, valid_mask=None) -> np.ndarray` (bool)
  box: filled boxes. diff: per box, gaussian blur sigma 2 of |gray diff|, Otsu
  inside the box, 5x5 close then open, keep components > 2% of box area; falls
  back to the filled box if nothing survives. diff without before_aligned -> box.
- `overlay(after, result, heat_alpha=0.35) -> np.ndarray` uint8 RGB: heat canvas
  blended as a colormap, mask tint, boxes coloured appeared=(0,220,0),
  disappeared=(230,40,40), changed=(255,200,0), label + score text, legend.
- `metrics(pred_mask, gt_mask, dets) -> dict` - pixel tp/fp/fn, iou, precision,
  recall, f1; gt_components, gt_components_hit (a component is hit if one box
  covers >= 50% of it or box/component-bbox IoU >= 0.3), component_recall;
  box_precision (fraction of boxes with >= 30% of their area on GT).
- `save_outputs(result, after, out_dir, *, save_heat=False) -> dict[str, str]` - writes
  `<id>_overlay.jpg`, `<id>_mask.png` (both at the canonical/working
  resolution), `<id>.json` (detections, windows, registration summary,
  timings, meta, `image_size` = working frame, `orig_image_size`, `scale`) and,
  with save_heat, `<id>_heat.png`. Each detection has `box` (canonical frame)
  and `box_orig` = box * result.meta["scale"] (original after-image pixels;
  equal to box when no scale is recorded). `Detection.to_dict(scale=None)`
  adds `box_orig` when given a scale.
  `<id>` has path-unsafe characters replaced by `_`.

## backends.py
- `class Backend` (base): `name`, `infer_stream(items) -> Iterator[(tag, out)]`
  where items yields `(tag, before_hwc, after_hwc)` and out is canonical
  (10,256,256) float32, results in FIFO order with tags preserved and asserted;
  `infer(before, after) -> out`; `close()`; `stats() -> dict`; context manager.
  `max_in_flight` bounds outstanding pushes. `stats()["latency"]` (mean/p50/
  p90/max ms per inference): asynchronous runtimes (neat, torch) measure from
  the start of the push to the output being in host memory (includes queueing
  behind earlier pushes; an upper bound when the caller pulls late);
  synchronous runtimes (`_SyncBackend`: onnx, afe, fake) report the blocking
  compute time of `_run` (subclasses override `_latency_ms`).
  `stats()["collect_wait"]`: time the caller was blocked in each collect/pull.
- `to_canonical(outputs) -> np.ndarray` - accepts the fused layout as
  (1,10,256,256) NCHW, (10,256,256), (256,256,10) HWC or (1,256,256,10), or the
  split layout [heat (2ch), wh (4ch), off (4ch)] in either NCHW or HWC; returns
  canonical (10,256,256) float32 (new array, never a view of a runtime buffer).
- `NeatBackend(model_path, *, use_graph=True, max_in_flight=2, timeout_ms=10000, measure=False)`
  pyneat: `Model(path, opt)` with `opt.preprocess.kind = InputKind.Tensor` when
  available; validates `input_specs()` = two inputs of (256,256,3) float32
  (accept rank-4 with batch 1); graph path: `g = pyneat.Graph();
  ro = pyneat.ModelRouteOptions(); ro.include_input = ro.include_output = True;
  g.add(model.graph(ro)); run = g.build([seed_b, seed_a])`; then
  `run.push_tensors([tb, ta])` / `run.pull_tensors(timeout)`; on any Graph
  failure falls back to `model.build([seed_b, seed_a])` (ModelRunner with
  push/pull). Tensors: `pyneat.Tensor.from_numpy(x, copy=True)` then
  `t.layout = pyneat.TensorLayout.HWC`. Copy every output with
  `np.array(t.to_numpy(), np.float32, copy=True)` immediately. All
  build/push/pull/close under `common.ACCELERATOR_LOCK`. `close_input()` then
  `close()` on shutdown. Optional `start_measurement()` report in `stats()`.
  `benchmark(num_samples=200)` -> dict from `Model.benchmark(BenchmarkOptions(...))`;
  it first shuts down the app's Run (Model.benchmark builds its own pipelines),
  so call it last. RuntimeErrors from push/pull/to_numpy are re-raised as
  CywsError (and mark the backend unusable). Flat outputs are reshaped by the
  output TensorSpec's `shape`.
- `OnnxBackend(onnx_path, providers=None)` - onnxruntime; feeds
  `in0_before`/`in1_after` as (1,3,256,256) NCHW; any output layout via to_canonical.
- `AfeBackend(model_name, model_dir)` - SiMa Model Compiler host execution (only
  inside the SDK container): `afe.apis.model.Model.load(name, dir)` then
  `execute({InputName("in0_before"): nhwc, ...}, fast_mode=True)`; see
  `AfeBackend` in app/backends.py for the exact calls.
- `FakeBackend(seed=0)` - deterministic, no model: heat peaks at blobs of large
  |before - after| difference with matching wh, for tests and dry runs.
- `make_backend(kind, model_path=None, **kw) -> Backend` for kind in
  neat | onnx | afe | fake.
- `cache_tag` (Neat/Onnx/Afe): `model_cache_tag(kind, path)` = `<kind>-<model file
  name without .tar.gz/.onnx/...>` (e.g. `neat-cyws_bf16_mpk`, `onnx-cyws_fused`),
  `afe-<model_name>`; eval/cached_backend.py uses it as the default cache namespace
  so different models never share cached outputs.

## app.py
- `class CywsApp(backend, cfg)`: `load(item) -> Pair` (load_pair with the
  config's squash/register policy, time stored in meta["load_ms"]);
  `process(pair) -> Result`:
  align (registration + warp, or plain resize) -> optional photometric match ->
  plan windows (drop windows with valid fraction < cfg.min_valid_frac) -> level 0
  first; if cfg.tiles == "gated" gate level >= 1 windows -> stream all windows
  through `backend.infer_stream` (both crops from `crop_pair`, same window) ->
  decode both branches -> filter_window -> drop_invalid -> merge per branch ->
  verify_detection (if cfg.verify_enabled()) -> fuse_branches -> make_mask ->
  heat canvas (max of both branches). Windows are planned in the canonical
  (working) frame, so tile auto = min(H, W) // 2 of it, a float tile is a
  fraction of its short side and an int `tile` is in working pixels. Crops (`crop_pair`) run in a
  `ThreadPoolExecutor(cfg.preprocess_threads)` (0 = inline), at most
  threads + max_in_flight windows ahead, consumed strictly in window order
  with tags (pair_id, window.id); gated mode streams level 0, gates, then
  streams the kept windows. result.meta carries `scale`, `orig_size`,
  `work_size`. `timings_ms` (ms, `TIMING_KEYS`):
  `load` (decode both images + GT mask), `register` (align: registration +
  warp or resize, photometric), `stream` (wall time of the overlapped
  crop + infer + decode loop), `preprocess_cpu` (summed worker crop time; CPU
  time, overlaps everything), `preprocess_wait` (main thread blocked on a
  crop), `infer_wait` (main thread blocked in the backend: push/pull, or a sync
  backend's compute), `decode` (CenterNet decode + filter + heat canvas, inside
  the stream), `postprocess` (everything outside align and stream: planning,
  gating, merge, verify, fuse, mask), `total`. Invariants:
  total = load + register + stream + postprocess (exact),
  stream ~= preprocess_wait + infer_wait + decode.
- `run_dataset(app, items, out_dir, *, log=print) -> dict` - processes every item,
  writes outputs and `summary.csv` / `metrics.json` (per pair + mean), returns
  the summary. Rows: working `width`/`height`, `orig_width`/`orig_height`,
  `t_<key>_ms` for every timing key, `mask_read_ms` (GT decode, inside t_load);
  metrics are computed in the working frame. CywsError fails one pair; any other exception propagates after
  a partial summary is written (`metrics.json` "aborted" names it).

## run_cyws_neat.py (CLI)
`--mode static|dataset`, `--before`, `--after`, `--mask`, `--dataset-dir`,
`--model-path`, `--backend neat|onnx|afe|fake` (default neat; inferred onnx for
`.onnx` paths), `--out`, `--preset fast|balanced|accurate` (default balanced;
`common.PRESETS`), `--register/--no-register`, `--reg-method`, `--photometric`,
`--verify-change/--no-verify-change`, `--levels 0,1`, `--tile auto|N|F` (N
working-frame pixels >= 64, F a fraction in (0, 1] of the short side),
`--overlap`, `--tiles full|gated`, `--gate-heat`, `--thr`, `--topk`, `--mask-mode box|diff`,
`--resize area|bilinear`, `--merge nms|wbf`, `--allow-squash`, `--save-heat`,
`--work-long N` (default 1920, 0 = native; AppConfig.work_long), `--threads N`
(default 4, 0 = inline; AppConfig.preprocess_threads),
`--limit N`, `--bench` (backend benchmark + per-stage timing table with the
stage sums, per-window stream / crop cpu / infer wait and the backend
latency), `-v`.
The preset sets levels / tile / tiles / gate_heat / thr / register; those
flags default to argparse.SUPPRESS, so any of them given explicitly overrides
the preset (`config_from_args`). `--preset balanced` with no other flags gives
exactly `AppConfig()`.
Validates paths before touching the accelerator; exits non-zero with a clear
message on CywsError.
