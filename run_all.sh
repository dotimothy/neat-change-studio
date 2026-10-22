#!/usr/bin/env bash
# Reproduce the CYWS Modalix port end to end, on the host.
#
#   BOARD=sima@<devkit-ip> ./run_all.sh   # everything, including the DevKit steps
#   SKIP_BOARD=1 ./run_all.sh             # host only (verify, export, compile, host checks)
#   STEPS="export compile" ./run_all.sh
#
# Steps: setup data fetch verify export probes calib compile reference board
# Needs: uv, docker with the NEAT SDK container (ghcr.io-sima-neat-sdk-latest),
# your before/after pairs in $CYWS_DATA (default data/testCYWS: images/<id>_ref.jpg,
# images/<id>_cap.jpg, optional masks/<id>.jpg), and for the board step
# passwordless ssh to $BOARD (user@devkit-ip). REMOTE is the install dir on the
# board (default /media/nvme/cyws). The data step downloads pairs from Google
# Drive only when TESTCYWS_MANIFEST names a manifest (see
# data/testcyws_manifest.example.tsv); otherwise it is skipped. The board step
# stops nothing by itself: free the MLA first (deploy.sh refuses while another
# MLA app is running unless FORCE=1).
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$PWD"
PY="$ROOT/.venv/bin/python"
REMOTE="${REMOTE:-/media/nvme/cyws}"
export CYWS_DATA="${CYWS_DATA:-$ROOT/data/testCYWS}"
STEPS="${STEPS:-setup data fetch verify export probes calib compile reference board}"
[[ -n "${SKIP_BOARD:-}" ]] && STEPS="${STEPS//board/}"
has() { [[ " $STEPS " == *" $1 "* ]]; }
say() { printf '\n=== %s\n' "$*"; }
if has board; then
  BOARD="${BOARD:?set BOARD=user@devkit-ip, e.g. BOARD=sima@<devkit-ip> ./run_all.sh (or SKIP_BOARD=1)}"
  export BOARD REMOTE
fi

if has setup; then
  say "host venv (.venv, reuses an existing torch install via --system-site-packages)"
  if [[ ! -x "$PY" ]]; then
    BASE_PY="${BASE_PY:-$(command -v python3)}"
    uv venv --python "$BASE_PY" --system-site-packages .venv
  fi
  uv pip install --python "$PY" -q gdown onnx onnxsim onnxruntime opencv-python-headless pillow scipy pytest einops \
    flask huggingface_hub
  "$PY" -c "import torch; print('torch', torch.__version__)"
fi
if has data; then
  if [[ -n "${TESTCYWS_MANIFEST:-}" ]]; then
    say "dataset (from Google Drive, manifest $TESTCYWS_MANIFEST)"
    data/download_testcyws.sh "$TESTCYWS_MANIFEST" "$CYWS_DATA"
  else
    say "dataset: TESTCYWS_MANIFEST is not set, skipping the download; using your pairs in $CYWS_DATA"
  fi
  if [[ -d "$CYWS_DATA/images" ]]; then "$PY" data/inspect_dataset.py "$CYWS_DATA"
  else echo "note: no pairs in $CYWS_DATA/images; verify, calib and the board dataset run need them" >&2; fi
fi
has fetch && { say "upstream CYWS code + checkpoint"; model/fetch_upstream.sh; }
has verify && { say "rewrite == upstream (float64 exact, folded, bf16 preview)"; "$PY" model/verify_reference.py; }
has export && { say "ONNX export (fused bf16 graph; split-output graph for int8 trials)"; "$PY" model/export_onnx.py --outputs fused; "$PY" model/export_onnx.py --outputs split; }
has probes && { say "probe compiles (risky patterns)"; "$PY" compile/probes/make_probes.py; compile/probes/run_probes.sh coam32 coam8 scse16_conv dec256 stem; }
has calib && { say "paired calibration / evaluation tensors"; "$PY" data/make_calib.py; }
if has compile; then
  say "compile bf16 (one MLA ELF) and the w8a16 variant"
  compile/compile.sh --precision bf16 --out build/bf16
  compile/compile.sh --model build/onnx/cyws_split.onnx --precision w8a16 --calib build/calib/calib_pairs.npz \
    --calib-method min_max --out build/w8a16_minmax
  "$PY" compile/mpk_io.py build/bf16/cyws_bf16_mpk.tar.gz
fi
has reference && { say "FP32 reference outputs (ONNX Runtime)"; "$PY" eval/run_model.py --backend onnx --model build/onnx/cyws_fused.onnx --out build/eval/fp32; }
if has board; then
  say "deploy + board evaluation"
  ./deploy.sh build/bf16/cyws_bf16_mpk.tar.gz --with-data
  ssh "$BOARD" "mkdir -p '$REMOTE/eval'"
  rsync -a eval/run_model.py build/calib/eval_inputs.npz "$BOARD:$REMOTE/eval/"
  ssh "$BOARD" "cd '$REMOTE' && source ~/pyneat/bin/activate && python3 eval/run_model.py --backend neat --model models/cyws_bf16_mpk.tar.gz --inputs eval/eval_inputs.npz --out eval/board_bf16 --benchmark 200"
  rsync -a "$BOARD:$REMOTE/eval/board_bf16.npy" "$BOARD:$REMOTE/eval/board_bf16.json" build/eval/
  "$PY" eval/compare.py build/eval/fp32 build/eval/board_bf16 --out build/eval/compare_board_bf16_vs_fp32.json
  ssh "$BOARD" "cd '$REMOTE' && ./run.sh --mode dataset --dataset-dir data/testCYWS --out out/bf16_default"
fi
say "done"
