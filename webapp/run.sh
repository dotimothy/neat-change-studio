#!/usr/bin/env bash
# Start the CYWS web demo on the DevKit. Extra arguments go to webapp/server.py:
#   webapp/run.sh                  # http(s)://<board>:8098/, bf16 model (open https:// for the webcam)
#   webapp/run.sh --model w8a16    # start with the faster build
#   webapp/run.sh --port 9000
#   webapp/run.sh --stop           # stop a running instance
#   webapp/run.sh --daemon         # start detached, log to webapp/server.log
#   webapp/run.sh --fetch bf16     # download a model from Hugging Face into models/ (or: all)
#   webapp/run.sh --list-hub       # what Hugging Face has, and what is here
#   webapp/run.sh --no-https       # plain HTTP only (the webcam then only works from localhost)
# Browsers allow the webcam only on HTTPS pages, so the same port also speaks HTTPS, with a
# self-signed certificate made in webapp/certs on first start; the browser warns about it
# once ("Advanced" -> continue).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
VENV="${PYNEAT_VENV:-$HOME/pyneat}"
PORT=8098
HTTPS=1
args=() daemon=0
# Model downloads don't serve anything or touch the MLA: run them straight away.
for a in "$@"; do
  if [[ "$a" == --fetch || "$a" == --list-hub ]]; then
    PY="python3"; [[ -f "${PYNEAT_VENV:-$HOME/pyneat}/bin/activate" ]] && PY="${PYNEAT_VENV:-$HOME/pyneat}/bin/python3"
    cd "$ROOT" && exec "$PY" "$HERE/server.py" "$@"
  fi
done
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stop)
      pid="$(pgrep -f "[w]ebapp/server.py" || true)"
      if [[ -n "$pid" ]]; then kill -TERM $pid; echo "stopped CYWS demo (pid $pid)"; else echo "CYWS demo is not running"; fi
      exit 0 ;;
    --daemon) daemon=1; shift ;;
    --port) PORT="$2"; args+=("$1" "$2"); shift 2 ;;
    --no-https) HTTPS=0; args+=("$1"); shift ;;
    *) args+=("$1"); shift ;;
  esac
done

if curl -fsS --max-time 1 "http://localhost:${PORT}/api/status" 2>/dev/null | grep -q '"cyws-demo"'; then
  echo "CYWS demo is already running at http://localhost:${PORT}/"; exit 0
fi
# One MLA user at a time (matches full command lines of other demos, not this script).
# The pattern lists command-line names of other MLA apps that may be running
# (bracketed first letters keep pgrep from matching itself); set
# MLA_BUSY_PATTERN to match your own.
MLA_BUSY_PATTERN="${MLA_BUSY_PATTERN:-[s]rc/app.py --config|[n]eat-genai-studio|[l]aya/|[c]ar-scratch|[m]la_client}"
busy="$(pgrep -af "$MLA_BUSY_PATTERN" || true)"
if [[ -n "$busy" && -z "${FORCE:-}" ]]; then
  echo "Another app may be using the MLA:" >&2; echo "$busy" >&2
  echo "Stop the other MLA app first, or start with FORCE=1 (only one process may use the MLA)" >&2
  exit 1
fi
[[ -f "$VENV/bin/activate" ]] || { echo "pyneat environment not found at $VENV" >&2; exit 1; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$ROOT"
export PYTHONUNBUFFERED=1
if ((HTTPS)); then
  CERTS="$HERE/certs"
  if [[ ! -f "$CERTS/cert.pem" || ! -f "$CERTS/key.pem" ]] && command -v openssl > /dev/null; then
    mkdir -p "$CERTS"
    san="DNS:localhost,DNS:$(hostname),IP:127.0.0.1"
    for ip in $(hostname -I); do san="$san,IP:$ip"; done
    openssl req -x509 -newkey rsa:2048 -nodes -days 825 -subj "/CN=CYWS demo" \
      -addext "subjectAltName=$san" -keyout "$CERTS/key.pem" -out "$CERTS/cert.pem" 2> /dev/null \
      && echo "made a self-signed certificate for $san"
  fi
fi
if ((daemon)); then
  setsid nohup python3 "$HERE/server.py" "${args[@]}" > "$HERE/server.log" 2>&1 < /dev/null &
  for _ in $(seq 1 60); do
    sleep 1
    if curl -fsS --max-time 1 "http://localhost:${PORT}/api/status" 2>/dev/null | grep -q '"state":"ready"'; then
      ip="$(hostname -I | awk '{print $1}')"
      echo "CYWS demo running at http://${ip}:${PORT}/  (log: $HERE/server.log)"
      ((HTTPS)) && echo "  webcam: https://${ip}:${PORT}/  (same port; self-signed: accept the browser warning once)"
      exit 0
    fi
  done
  echo "started, but the model is not ready yet; see $HERE/server.log" >&2; exit 0
fi
exec python3 "$HERE/server.py" "${args[@]}"
