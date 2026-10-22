#!/usr/bin/env bash
# Run compile/cyws_compile.py inside the NEAT SDK container's Model Compiler venv.
#
#   compile/compile.sh --precision bf16
#   compile/compile.sh --precision int8 --calib build/calib/calib_pairs.npz --calib-method min_max --out build/int8_minmax
#   compile/compile.sh --model build/probes/coam32.onnx --precision bf16 --out build/probes/coam32
#
# Output goes to build/<precision>/ (or --out), with the full compiler log next
# to it as compile.log.
#
# The script runs `docker exec` in an already running NEAT SDK container:
#   CONTAINER  name of that container (default ghcr.io-sima-neat-sdk-latest, the
#              name the SDK installer gives it); it must have the Model Compiler
#              at /sdk-extensions/model-compiler.
#   WS_HOST    host directory the container mounts at /workspace (default
#              ~/SDK/NEAT/workspace). This repo must be inside it, so that the
#              container sees it at /workspace/<path below WS_HOST>.
#
# XLA_FLAGS: the compiler evaluates the model with JAX/XLA on the host to write
# the MLA check file. On some recent x86 CPUs, XLA's LLVM vectorizer emits bf16
# code it cannot lower ("LLVM ERROR: Cannot select: v8bf16 = X86ISD::BLENDV"),
# and every bf16 compile with a Conv fails with "Failed to generate mpk json
# file". Turning the vectorizers off for that host-side step avoids it; the MLA
# program is unaffected. Set XLA_FLAGS yourself to override.
set -euo pipefail
XLA_FLAGS="${XLA_FLAGS:---xla_backend_extra_options=-vectorize-loops=false,-vectorize-slp=false}"

HERE="$(cd "$(dirname "$0")/.." && pwd -P)"
CONTAINER="${CONTAINER:-ghcr.io-sima-neat-sdk-latest}"
WS_HOST="$(cd "${WS_HOST:-$HOME/SDK/NEAT/workspace}" && pwd -P)"
REL="${HERE#"$WS_HOST"/}"
[[ "$REL" != "$HERE" ]] || { echo "$HERE is not under $WS_HOST (the container's /workspace; set WS_HOST)" >&2; exit 1; }

OUT="" PREC="bf16" prev=""
for a in "$@"; do
  [[ "$prev" == "--out" ]] && OUT="$a"
  [[ "$prev" == "--precision" ]] && PREC="$a"
  prev="$a"
done
OUT="${OUT:-build/$PREC}"
mkdir -p "$HERE/$OUT"

args=""
for a in "$@"; do args+=" $(printf '%q' "$a")"; done
set +e
docker exec -w "/workspace/$REL" -e HOME=/tmp -e "XLA_FLAGS=$XLA_FLAGS" \
  -e "SIMA_MLA_SIM_PARALLEL=${SIMA_MLA_SIM_PARALLEL:-8}" -u "$(id -u):$(id -g)" "$CONTAINER" bash -c \
  "source /sdk-extensions/model-compiler/bin/activate && python3 -u compile/cyws_compile.py$args" \
  2>&1 | tee "$HERE/$OUT/compile.log"
rc=${PIPESTATUS[0]}
echo "compile exit code $rc  (log: $OUT/compile.log)"
exit "$rc"
