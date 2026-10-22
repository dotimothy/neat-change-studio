#!/usr/bin/env python3
"""CYWS change detection on Modalix: command-line entry point.

    python3 run_cyws_neat.py --mode static --before b.jpg --after a.jpg --model-path cyws.tar.gz
    python3 run_cyws_neat.py --mode dataset --dataset-dir data/testCYWS --preset fast

Presets (--preset, default balanced) set levels / tile / tiles / gate-heat / thr /
register; any of those flags given explicitly overrides the preset.

Every path and option is validated before a backend is created, so a typo never
costs an accelerator build. The module imports neither pyneat nor onnxruntime;
backends import them lazily.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from common import PRESETS, AppConfig, CywsError, PairItem

BACKENDS = ("neat", "onnx", "afe", "fake")


def _levels(text: str) -> tuple[int, ...]:
    try:
        levels = tuple(int(v) for v in text.split(",") if v.strip() != "")
    except ValueError:
        raise argparse.ArgumentTypeError(f"--levels expects comma-separated integers, got {text!r}") from None
    if not levels or any(lv not in (0, 1, 2) for lv in levels) or len(set(levels)) != len(levels):
        raise argparse.ArgumentTypeError(f"--levels expects distinct values from 0,1,2, got {text!r}")
    return tuple(sorted(levels))


def _tile(text: str) -> int | float | str:
    """'auto', an integer >= 64 (working-frame pixels) or a fraction in (0, 1] of the short side."""
    if text == "auto":
        return "auto"
    try:
        value = int(text)
    except ValueError:
        try:
            frac = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"--tile expects 'auto', an integer or a fraction, got {text!r}") from None
        if not 0.0 < frac <= 1.0:
            raise argparse.ArgumentTypeError(f"--tile as a fraction must be in (0, 1], got {frac}")
        return frac
    if value < 64:
        raise argparse.ArgumentTypeError(f"--tile must be >= 64 pixels, got {value}")
    return value


# Flags a preset sets. They default to SUPPRESS (absent from the namespace) so
# config_from_args can tell "not given" (take the preset) from an explicit value.
PRESET_FLAGS = ("levels", "tile", "tiles", "gate_heat", "thr", "register")


def _preset_help(field: str) -> str:
    vals = "; ".join(f"{name}: {_fmt_value(cfg[field])}" for name, cfg in PRESETS.items())
    return f"(default from --preset: {vals})"


def _fmt_value(v: Any) -> str:
    if isinstance(v, tuple):
        return ",".join(str(x) for x in v)
    return str(v)


def _unit_float(lo: float, hi: float, name: str, *, lo_open: bool = False, hi_open: bool = False):
    def parse(text: str) -> float:
        try:
            v = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{name} expects a number, got {text!r}") from None
        if (v <= lo if lo_open else v < lo) or (v >= hi if hi_open else v > hi):
            raise argparse.ArgumentTypeError(f"{name} must be in {'(' if lo_open else '['}{lo}, {hi}"
                                             f"{')' if hi_open else ']'}, got {v}")
        return v

    return parse


def _positive_int(name: str):
    def parse(text: str) -> int:
        try:
            v = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{name} expects an integer, got {text!r}") from None
        if v < 1:
            raise argparse.ArgumentTypeError(f"{name} must be >= 1, got {v}")
        return v

    return parse


def _nonneg_int(name: str):
    def parse(text: str) -> int:
        try:
            v = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{name} expects an integer, got {text!r}") from None
        if v < 0:
            raise argparse.ArgumentTypeError(f"{name} must be >= 0, got {v}")
        return v

    return parse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_cyws_neat.py",
        description="Detect changes between a before and an after image with CYWS on the Modalix MLA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    io = p.add_argument_group("input / output")
    io.add_argument("--mode", choices=("static", "dataset"), default="static")
    io.add_argument("--before", type=Path, help="before (reference) image, static mode")
    io.add_argument("--after", type=Path, help="after (query) image, static mode; defines the output frame")
    io.add_argument("--mask", type=Path, help="optional ground-truth change mask, static mode")
    io.add_argument("--dataset-dir", type=Path, help="dataset root with images/<id>_ref|_cap and masks/<id>")
    io.add_argument("--limit", type=_positive_int("--limit"), help="process only the first N dataset pairs")
    io.add_argument("--out", type=Path, default=Path("out"), help="output directory")
    io.add_argument("--save-heat", action="store_true", help="also write <id>_heat.png")
    io.add_argument("--allow-squash", action="store_true",
                    help="resize a before image with a different aspect ratio instead of failing")

    be = p.add_argument_group("backend")
    be.add_argument("--model-path", type=Path, help="compiled MPK (.tar.gz) or .onnx / afe model directory")
    be.add_argument("--backend", choices=BACKENDS, default=None,
                    help="default: onnx for .onnx model paths, else neat")
    be.add_argument("--bench", action="store_true", help="backend benchmark + per-stage timing table")
    be.add_argument("-v", "--verbose", action="store_true")

    pr = p.add_argument_group("preset")
    pr.add_argument("--preset", choices=tuple(PRESETS), default="balanced",
                    help="fast: level 0 only (1 window per pair); balanced: + level-1 tiles where level 0 saw "
                         "heat; accurate: + every level-1 tile. Explicit flags override the preset")

    S = argparse.SUPPRESS
    rg = p.add_argument_group("registration / false-positive filtering")
    rg.add_argument("--register", action=argparse.BooleanOptionalAction, default=S,
                    help="warp before into the after frame (SIFT homography); for handheld captures, costs "
                         "SIFT time on the CPU " + _preset_help("register"))
    rg.add_argument("--reg-method", choices=("sift", "orb"), default="sift")
    rg.add_argument("--photometric", action="store_true", help="histogram-match before to after")
    rg.add_argument("--verify-change", action=argparse.BooleanOptionalAction, default=None,
                    help="drop boxes whose patches still match after alignment (default: on iff --register)")

    tl = p.add_argument_group("tiling")
    tl.add_argument("--levels", type=_levels, default=S,
                    help="pyramid levels, e.g. 0 or 0,1 or 0,1,2 " + _preset_help("levels"))
    tl.add_argument("--tile", type=_tile, default=S,
                    help="level-1 window side: a fraction in (0, 1] of the working frame's short side, an "
                         "integer in working-frame pixels (see --work-long), or auto = min(H, W) // 2 "
                         + _preset_help("tile"))
    tl.add_argument("--overlap", type=_unit_float(0.0, 0.95, "--overlap", hi_open=True), default=0.25)
    tl.add_argument("--tiles", choices=("full", "gated"), default=S,
                    help="gated: run level >= 1 windows only where level 0 saw heat >= --gate-heat "
                         + _preset_help("tiles"))
    tl.add_argument("--gate-heat", type=_unit_float(0.0, 1.0, "--gate-heat"), default=S,
                    help="gated tiles: level-0 heat a level >= 1 window needs to run " + _preset_help("gate_heat"))
    tl.add_argument("--resize", choices=("area", "bilinear"), default="area")

    cpu = p.add_argument_group("decode / CPU")
    cpu.add_argument("--work-long", type=_nonneg_int("--work-long"), default=1920,
                     help="decode so the after image's long side is >= this and its short side >= half of it "
                          "(power-of-two reduction, JPEG draft decoding); boxes, tiles and metrics live in that working frame; 0 = native")
    cpu.add_argument("--threads", type=_nonneg_int("--threads"), default=4,
                     help="crop worker threads overlapping the accelerator; 0 = crop in the main thread")

    pp = p.add_argument_group("decode / merge / mask")
    pp.add_argument("--thr", type=_unit_float(0.0, 1.0, "--thr", lo_open=True), default=S,
                    help="detection score threshold " + _preset_help("thr"))
    pp.add_argument("--topk", type=_positive_int("--topk"), default=100)
    pp.add_argument("--merge", choices=("nms", "wbf"), default="nms")
    pp.add_argument("--mask-mode", choices=("box", "diff"), default="box")
    return p


def config_from_args(args: argparse.Namespace) -> AppConfig:
    """Map parsed CLI options 1:1 onto AppConfig fields.

    Start from the --preset; each preset flag given explicitly overrides it.
    """
    preset = dict(PRESETS[getattr(args, "preset", "balanced")])
    for name in PRESET_FLAGS:
        if hasattr(args, name):
            preset[name] = getattr(args, name)
    preset["levels"] = tuple(preset["levels"])
    return AppConfig(
        **preset,
        overlap=args.overlap,
        resize=args.resize,
        work_long=args.work_long,
        preprocess_threads=args.threads,
        topk=args.topk,
        merge=args.merge,
        reg_method=args.reg_method,
        photometric=args.photometric,
        verify_change=args.verify_change,
        mask_mode=args.mask_mode,
        allow_squash=args.allow_squash,
        save_heat=args.save_heat,
    )


def resolve_backend(args: argparse.Namespace) -> str:
    """Pick the backend and check its model path exists, without loading anything."""
    kind = args.backend
    if kind is None:
        kind = "onnx" if args.model_path is not None and args.model_path.suffix.lower() == ".onnx" else "neat"
    if kind == "fake":
        return kind
    if args.model_path is None:
        raise CywsError(f"--model-path is required for the {kind} backend")
    path: Path = args.model_path
    if not path.exists():
        raise CywsError(f"model not found: {path}")
    if kind in ("neat", "onnx") and not path.is_file():
        raise CywsError(f"{kind} backend expects a model file, got a directory: {path}")
    if kind == "neat" and not path.name.endswith((".tar.gz", ".tgz", ".mpk")):
        raise CywsError(f"neat backend expects a compiled MPK (*_mpk.tar.gz), got {path.name}")
    if kind == "onnx" and path.suffix.lower() != ".onnx":
        raise CywsError(f"onnx backend expects a .onnx file, got {path.name}")
    return kind


def collect_items(args: argparse.Namespace, log) -> list[PairItem]:
    """Validate the input paths for the chosen mode and return the pairs to run."""
    import dataset_loader

    if args.mode == "static":
        if args.before is None or args.after is None:
            raise CywsError("static mode needs --before and --after")
        if args.dataset_dir is not None:
            raise CywsError("--dataset-dir is only used in dataset mode")
        return [dataset_loader.static_pair(args.before, args.after, args.mask)]
    if args.dataset_dir is None:
        raise CywsError("dataset mode needs --dataset-dir")
    if args.before is not None or args.after is not None or args.mask is not None:
        raise CywsError("--before/--after/--mask are only used in static mode")
    items, warnings = dataset_loader.discover_pairs(args.dataset_dir)
    for w in warnings:
        log(f"warning: {w}")
    if args.limit is not None:
        items = items[: args.limit]
    return items


def make_backend(kind: str, model_path: Optional[Path], cfg: AppConfig, *, measure: bool = False) -> Any:
    """Create the backend; imported here so validation never loads a runtime.

    measure=True (from --bench) turns on pyneat's per-stage measurement so the
    final backend stats carry the runtime's own latency breakdown.
    """
    import backends

    kw: dict[str, Any] = {}
    if kind == "neat":
        kw["max_in_flight"] = cfg.max_in_flight
        kw["measure"] = measure
    return backends.make_backend(kind, str(model_path) if model_path is not None else None, **kw)


def _stage_table(summary: dict) -> str:
    """Per-stage latency table (mean / p50 / p90 / max ms) over the successful pairs.

    Wall-clock stages add up: total = load + register + stream + postprocess,
    and stream ~= preprocess_wait + infer_wait + decode. preprocess_cpu is the
    summed worker-thread crop time (overlapped, so it can exceed stream).
    """
    rows = [r for r in summary.get("pairs", []) if r.get("status") == "ok"]
    if not rows:
        return "no successful pairs to time"
    keys = [k for k in rows[0] if k.startswith("t_") and k.endswith("_ms")]
    lines = [f"{'stage':<16}{'mean':>10}{'p50':>10}{'p90':>10}{'max':>10}   (ms over {len(rows)} pairs)"]
    for k in keys:
        v = np.array([float(r[k]) for r in rows if isinstance(r.get(k), (int, float))])
        if v.size:
            lines.append(f"{k[2:-3]:<16}{v.mean():>10.1f}{np.percentile(v, 50):>10.1f}"
                         f"{np.percentile(v, 90):>10.1f}{v.max():>10.1f}")
    lines.append("total = load + register + stream + postprocess; "
                 "stream ~= preprocess_wait + infer_wait + decode; preprocess_cpu = summed worker time")
    wins = np.array([float(r.get("windows", 0)) for r in rows])
    if wins.sum() > 0:
        per = {k: np.array([float(r.get(f"t_{k}_ms", 0.0)) for r in rows]).sum() / wins.sum()
               for k in ("stream", "preprocess_cpu", "infer_wait")}
        lines.append(f"windows/pair {wins.mean():.1f}; per window: stream {per['stream']:.1f} ms, "
                     f"crop cpu {per['preprocess_cpu']:.1f} ms, infer wait {per['infer_wait']:.1f} ms")
    lat = (summary.get("backend_stats") or {}).get("latency") or {}
    if lat:
        lines.append(f"backend latency per inference: mean {lat.get('mean_ms', 0):.1f} ms, "
                     f"p50 {lat.get('p50_ms', 0):.1f} ms, max {lat.get('max_ms', 0):.1f} ms")
    return "\n".join(lines)


def _print_bench(backend: Any, summary: dict, log) -> None:
    log("")
    log(_stage_table(summary))
    bench = getattr(backend, "benchmark", None)
    if callable(bench):
        from common import ACCELERATOR_LOCK

        try:
            with ACCELERATOR_LOCK:
                report = bench(num_samples=200)
            log("backend benchmark: " + ", ".join(f"{k}={v}" for k, v in dict(report).items()))
        except Exception as e:  # benchmark support varies by runtime build; the run itself succeeded
            log(f"backend benchmark unavailable: {type(e).__name__}: {e}")
    stats = getattr(backend, "stats", None)
    if callable(stats):
        st = stats()
        if st:
            log("backend stats: " + ", ".join(f"{k}={v}" for k, v in dict(st).items()))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    log = print
    try:
        cfg = config_from_args(args)
        from app import CywsApp, run_dataset, validate_config

        validate_config(cfg)
        kind = resolve_backend(args)
        items = collect_items(args, log)
        if not items:
            raise CywsError("nothing to process")
        if args.verbose:
            log(f"backend={kind} pairs={len(items)} preset={args.preset} levels={cfg.levels} tile={cfg.tile} "
                f"tiles={cfg.tiles} gate_heat={cfg.gate_heat} thr={cfg.thr} "
                f"register={cfg.register} verify={cfg.verify_enabled()} mask={cfg.mask_mode} "
                f"work_long={cfg.work_long or 'native'} threads={cfg.preprocess_threads} out={args.out}")
        backend = make_backend(kind, args.model_path, cfg, measure=args.bench)
        try:
            app = CywsApp(backend, cfg)
            summary = run_dataset(app, items, args.out, log=log)
            if args.mode == "static" and args.verbose and summary["n_ok"]:
                import json

                # save_outputs sanitises the pair id, so take the path it actually wrote.
                files = str(summary["pairs"][0].get("files", "")).split(";")
                json_path = next((f for f in files if f.endswith(".json")), None)
                if json_path:
                    log(json.dumps(json.loads(Path(json_path).read_text()).get("detections", []), indent=1))
            if args.bench:
                _print_bench(backend, summary, log)
        finally:
            backend.close()
    except CywsError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130

    pooled = summary.get("pooled") or {}
    log(f"done: {summary['n_ok']}/{summary['n_pairs']} pairs ok -> {args.out}"
        + (f" | pooled IoU {pooled['iou']:.3f} F1 {pooled['f1']:.3f}" if pooled else ""))
    if summary["n_ok"] == 0:
        print("error: no pair was processed successfully", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
