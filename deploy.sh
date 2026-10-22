#!/usr/bin/env bash
# Copy the CYWS app, web demo, scripts and (optionally) a compiled MPK to the
# DevKit, then run setup.sh there. BOARD (user@devkit-ip) is required.
#   BOARD=sima@<devkit-ip> ./deploy.sh                   # app + web demo only
#   BOARD=sima@<devkit-ip> ./deploy.sh build/bf16/cyws_bf16_mpk.tar.gz   # also an MPK
#   BOARD=sima@<devkit-ip> MPK=build/bf16/cyws_bf16_mpk.tar.gz ./deploy.sh
#   BOARD=sima@<devkit-ip> ./deploy.sh --with-data       # also copy $DATA (your pairs)
#   BOARD=sima@<devkit-ip> REMOTE=/media/nvme/cyws FORCE=1 ./deploy.sh
# REMOTE defaults to /media/nvme/cyws. DATA defaults to $CYWS_DATA, or data/testCYWS
# in the repo. Without an MPK, fetch a model on the board from Hugging Face with
# webapp/run.sh --fetch bf16. Refuses while another MLA app runs unless FORCE=1.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
for a in "$@"; do [[ "$a" == -h || "$a" == --help ]] && { sed -n '2,12p' "$0"; exit 0; }; done
BOARD="${BOARD:?set BOARD=user@devkit-ip, e.g. BOARD=sima@<devkit-ip> ./deploy.sh}"
REMOTE="${REMOTE:-/media/nvme/cyws}"
DATA="${DATA:-${CYWS_DATA:-$HERE/data/testCYWS}}"
MPK="${MPK:-}"
WITH_DATA=0
for a in "$@"; do
  case "$a" in
    --with-data) WITH_DATA=1 ;;
    -*) echo "unknown option: $a" >&2; exit 2 ;;
    *) MPK="$a" ;;
  esac
done

if [[ -n "$MPK" ]]; then
  [[ -f "$MPK" ]] || { echo "MPK not found: $MPK" >&2; exit 1; }
  [[ "$MPK" == *.tar.gz ]] || { echo "expected a compiled *_mpk.tar.gz, got $MPK" >&2; exit 1; }
fi
if ((WITH_DATA)); then
  [[ -d "$DATA/images" ]] || { echo "dataset not found: $DATA/images" >&2; exit 1; }
fi
for f in app/run_cyws_neat.py app/app.py app/common.py run.sh setup.sh webapp/run.sh webapp/server.py; do
  [[ -f "$HERE/$f" ]] || { echo "missing $HERE/$f" >&2; exit 1; }
done

SSH=(ssh -o ConnectTimeout=10)
# One MLA user at a time: warn about anything that may hold the accelerator.
# The pattern lists command-line names of other MLA apps that may be running on
# the board (bracketed first letters keep pgrep from matching itself); set
# MLA_BUSY_PATTERN to match your own.
MLA_BUSY_PATTERN="${MLA_BUSY_PATTERN:-[n]eat-genai-studio|[l]aya|[c]ar-scratch|[d]ofa|[l]lima}"
busy="$("${SSH[@]}" "$BOARD" "pgrep -af $(printf '%q' "$MLA_BUSY_PATTERN") || true")"
if [[ -n "$busy" ]]; then
  echo "warning: another app may be using the MLA on $BOARD:" >&2; echo "$busy" >&2
  if [[ -z "${FORCE:-}" ]]; then
    echo "Stop it before running CYWS, or deploy anyway with FORCE=1 ./deploy.sh" >&2
    exit 1
  fi
fi

"${SSH[@]}" "$BOARD" "mkdir -p '$REMOTE/app' '$REMOTE/models' '$REMOTE/out'"
rsync -a --delete --exclude __pycache__ --exclude '*.pyc' -e "${SSH[*]}" "$HERE/app/" "$BOARD:$REMOTE/app/"
rsync -a --delete --exclude __pycache__ -e "${SSH[*]}" "$HERE/bench/" "$BOARD:$REMOTE/bench/"
# The web demo; keeps the board's own certificates, uploads, cache and log.
rsync -a --delete --exclude __pycache__ --exclude '*.pyc' --exclude /certs --exclude /uploads \
  --exclude /cache --exclude /server.log -e "${SSH[*]}" "$HERE/webapp/" "$BOARD:$REMOTE/webapp/"
extra=("$HERE/run.sh" "$HERE/setup.sh")
[[ -f "$HERE/README.md" ]] && extra+=("$HERE/README.md") || echo "note: no README.md yet, skipping"
rsync -a -e "${SSH[*]}" "${extra[@]}" "$BOARD:$REMOTE/"
if [[ -n "$MPK" ]]; then
  rsync -a --checksum -e "${SSH[*]}" "$MPK" "$BOARD:$REMOTE/models/$(basename "$MPK")"
  echo "model -> $REMOTE/models/$(basename "$MPK")"
else
  echo "note: no MPK given (MPK=path or first argument); deploying the app only."
  echo "      On the board, fetch a model from Hugging Face with: $REMOTE/webapp/run.sh --fetch bf16"
fi
if ((WITH_DATA)); then
  "${SSH[@]}" "$BOARD" "mkdir -p '$REMOTE/data/testCYWS'"
  rsync -a -e "${SSH[*]}" "$DATA/" "$BOARD:$REMOTE/data/testCYWS/"
  echo "dataset -> $REMOTE/data/testCYWS"
fi
"${SSH[@]}" "$BOARD" "chmod +x '$REMOTE'/*.sh && '$REMOTE/setup.sh'" || echo "warning: setup.sh reported problems on the board" >&2

echo
echo "deployed to $BOARD:$REMOTE. On the board:"
echo "  $REMOTE/run.sh --mode dataset --dataset-dir $REMOTE/data/testCYWS --levels 0,1"
echo "  $REMOTE/webapp/run.sh --daemon      # web demo on port 8098"
