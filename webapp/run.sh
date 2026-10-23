#!/usr/bin/env bash
# Start the Neat Change Studio web demo on the DevKit (CYWS and DINOv2 models).
# Extra arguments go to webapp/server.py:
#   webapp/run.sh                  # http(s)://<board>:8098/, bf16 model (open https:// for the webcam)
#   webapp/run.sh --model w8a16    # start with the faster build (or dinov2-448, zero-shot DINOv2)
#   webapp/run.sh --port 9000
#   webapp/run.sh --stop           # stop a running instance
#   webapp/run.sh --daemon         # start detached, log to webapp/server.log
#   webapp/run.sh --fetch bf16     # download a model from Hugging Face into models/ (or: w8a16, dinov2-448, all)
#   webapp/run.sh --hub-repo dotimothy/sima-cyws   # read only these Hugging Face repos ('' = none)
#   webapp/run.sh --list-hub       # what Hugging Face has, and what is here
#   webapp/run.sh --verify all     # check the models here against their published SHA-256
#   webapp/run.sh --delete w8a16   # delete a model (not the one the running demo has loaded)
#   webapp/run.sh --no-https       # plain HTTP only (the webcam then only works from localhost)
#   webapp/run.sh update           # fetch the latest source over this install, then re-run setup.sh
# Browsers allow the webcam only on HTTPS pages, so the same port also speaks HTTPS, with a
# self-signed certificate made in webapp/certs on first start (webapp/make_cert.sh); the
# browser warns about it once ("Advanced" -> continue).
# setup.sh can add a 'neat-change' shell alias for this script.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"

# `update`: the latest install.sh fetches the latest source (STUDIO_REPO, STUDIO_BRANCH,
# GITHUB_TOKEN and STUDIO_ARCHIVE work as for a fresh install) and copies it over this install,
# keeping models/, data/, out/, webapp/certs, webapp/uploads, webapp/cache and the other files
# .gitignore lists; then setup.sh runs without prompts. install.sh itself is downloaded first,
# so a fix to it applies to this update; the local copy is used with STUDIO_ARCHIVE (an
# offline board) or when the download fails. The whole block is read before it runs, so this
# file can change underneath it.
if [[ "${1:-}" == update ]]; then
  shift
  repo="${STUDIO_REPO:-dotimothy/neat-change-studio}" branch="${STUDIO_BRANCH:-main}"
  installer="$(mktemp)" || exit 1
  trap 'rm -f "$installer"' EXIT
  fetch_installer() {   # the branch's install.sh into $installer
    local url="https://raw.githubusercontent.com/${repo}/${branch}/install.sh"
    local -a auth=()
    if [[ -n "${GITHUB_TOKEN:-}" ]]; then   # a private repository: the API takes the token
      url="https://api.github.com/repos/${repo}/contents/install.sh?ref=${branch}"
      auth=(-H "Authorization: Bearer ${GITHUB_TOKEN}" -H "Accept: application/vnd.github.raw")
    fi
    if command -v curl > /dev/null 2>&1; then
      curl -fsSL --max-time 60 ${auth[@]+"${auth[@]}"} "$url" -o "$installer" 2> /dev/null || return 1
    elif command -v wget > /dev/null 2>&1; then
      wget -qO "$installer" --timeout=60 ${GITHUB_TOKEN:+--header="Authorization: Bearer ${GITHUB_TOKEN}"} \
        ${GITHUB_TOKEN:+--header="Accept: application/vnd.github.raw"} "$url" || return 1
    else
      return 1
    fi
    grep -q 'STUDIO_UPDATE' "$installer"   # the script, not an error page
  }
  if [[ -z "${STUDIO_ARCHIVE:-}" ]] && fetch_installer; then
    echo "Using the latest install.sh (${repo}, ${branch})"
  elif [[ -f "$ROOT/install.sh" ]]; then
    [[ -z "${STUDIO_ARCHIVE:-}" ]] && echo "Could not download install.sh; using the one in $ROOT"
    cp "$ROOT/install.sh" "$installer" || exit 1
  elif [[ -n "${STUDIO_ARCHIVE:-}" ]] && fetch_installer; then
    echo "Using the latest install.sh (${repo}, ${branch})"
  else
    echo "Could not download install.sh and there is none in $ROOT." >&2
    echo "Get it from https://github.com/${repo} (or check the board's internet access)." >&2
    exit 1
  fi
  STUDIO_UPDATE=1 STUDIO_SETUP=0 STUDIO_DIR="$ROOT" bash "$installer" || exit $?
  echo
  "$ROOT/setup.sh" --no-prompt "$@" || exit $?
  if pgrep -f "$HERE/server.py" > /dev/null 2>&1; then
    echo
    echo "The web demo from this install is still running the old code. Restart it with:"
    echo "  $HERE/run.sh --stop && $HERE/run.sh --daemon"
  fi
  exit 0
fi
VENV="${PYNEAT_VENV:-$HOME/pyneat}"
PORT=8098
HTTPS=1
args=() daemon=0
# Model downloads don't serve anything or touch the MLA: run them straight away.
for a in "$@"; do
  if [[ "$a" == --fetch || "$a" == --list-hub || "$a" == --delete || "$a" == --verify ]]; then
    PY="python3"; [[ -f "${PYNEAT_VENV:-$HOME/pyneat}/bin/activate" ]] && PY="${PYNEAT_VENV:-$HOME/pyneat}/bin/python3"
    cd "$ROOT" && exec "$PY" "$HERE/server.py" "$@"
  fi
done
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stop)
      pid="$(pgrep -f "[w]ebapp/server.py" || true)"
      if [[ -n "$pid" ]]; then
        # shellcheck disable=SC2086
        kill -TERM $pid
        # wait until it has let go of the MLA, so a --daemon straight after does not find it busy
        for _ in $(seq 1 40); do
          alive=0; for p in $pid; do kill -0 "$p" 2>/dev/null && alive=1; done
          ((alive)) || break; sleep 0.5
        done
        ((alive)) && echo "CYWS demo (pid $pid) is still shutting down" >&2 || echo "stopped CYWS demo (pid $pid)"
      else echo "CYWS demo is not running"; fi
      exit 0 ;;
    --daemon) daemon=1; shift ;;
    --port) [[ $# -ge 2 ]] || { echo "--port needs a value" >&2; exit 2; }
            PORT="$2"; args+=("$1" "$2"); shift 2 ;;
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
# MLA_BUSY_PATTERN to match your own. setup.sh --check uses the same default; keep the two
# alike. A demo from this install on this port was found above; [w]ebapp/server.py here is
# one on another port.
MLA_BUSY_PATTERN="${MLA_BUSY_PATTERN:-[n]eat-genai-studio|[l]aya/|[c]ar-scratch|[d]ofa|[l]lima|[s]rc/app.py --config|[m]la_client|[w]ebapp/server.py|[r]un_cyws_neat.py}"
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
  bash "$HERE/make_cert.sh" "$HERE/certs" || true
fi
if ((daemon)); then
  setsid nohup python3 "$HERE/server.py" "${args[@]}" > "$HERE/server.log" 2>&1 < /dev/null &
  for _ in $(seq 1 60); do
    sleep 1
    if curl -fsS --max-time 1 "http://localhost:${PORT}/api/status" 2>/dev/null | grep -q '"state":"ready"'; then
      ip="$(hostname -I 2> /dev/null | awk '{print $1}' || true)"; ip="${ip:-localhost}"
      echo "CYWS demo running at http://${ip}:${PORT}/  (log: $HERE/server.log)"
      ((HTTPS)) && echo "  webcam: https://${ip}:${PORT}/  (same port; self-signed: accept the browser warning once)"
      exit 0
    fi
  done
  echo "started, but the model is not ready yet; see $HERE/server.log" >&2; exit 0
fi
exec python3 "$HERE/server.py" "${args[@]}"
