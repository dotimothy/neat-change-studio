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
MODEL=bf16
args=() daemon=0
# shellcheck source=webapp/term.sh
source "$HERE/term.sh"
# shellcheck source=webapp/models_dir.sh
source "$HERE/models_dir.sh"
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
      step "Stopping Neat Change Studio"
      # only this install's server: a python process whose script is $HERE/server.py (not another
      # app's webapp/server.py, such as Neat DINO Studio's, nor a shell whose command line names it)
      pid="$(pgrep -f "^[^ ]*python[0-9.]* $HERE/server\.py( |\$)" || true)"
      if [[ -n "$pid" ]]; then
        # shellcheck disable=SC2086
        kill -TERM $pid
        # wait until it has let go of the MLA, so a --daemon straight after does not find it busy
        for _ in $(seq 1 40); do
          alive=0; for p in $pid; do kill -0 "$p" 2>/dev/null && alive=1; done
          ((alive)) || break; sleep 0.5
        done
        if ((alive)); then warn "Still shutting down (pid ${pid//$'\n'/ })"
        else ok "Stopped (pid ${pid//$'\n'/ }); the MLA is free"; fi
      else info "It is not running"; fi
      exit 0 ;;
    --daemon) daemon=1; shift ;;
    --port) [[ $# -ge 2 ]] || { echo "--port needs a value" >&2; exit 2; }
            PORT="$2"; args+=("$1" "$2"); shift 2 ;;
    --model) [[ $# -ge 2 ]] || { echo "--model needs a value" >&2; exit 2; }
             MODEL="$2"; args+=("$1" "$2"); shift 2 ;;
    --model=*) MODEL="${1#*=}"; args+=("$1"); shift ;;
    --no-https) HTTPS=0; args+=("$1"); shift ;;
    *) args+=("$1"); shift ;;
  esac
done

title_of() {   # a model's name as the app shows it
  case "$1" in
    bf16) echo "CYWS bf16" ;; w8a16) echo "CYWS w8a16" ;; dinov2-448) echo "DINOv2 448" ;; *) echo "$1" ;;
  esac
}
status_json() { curl -fsS --max-time 1 "http://localhost:${PORT}/api/status" 2>/dev/null; }
# One line from /api/status: state, load stage, the loaded (or loading) model's title, error (\x1f between).
status_line() {
  python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(1)
titles = {m.get("name"): m.get("title") or m.get("name") for m in d.get("models") or []}
load = d.get("load") or {}
name = d.get("model") or load.get("target") or ""
print("\x1f".join([d.get("state") or "", load.get("stage") or "", titles.get(name, name), d.get("error") or ""]).replace("\n", " "))
' 2>/dev/null
}
IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"; IP="${IP:-localhost}"
URL="http://${IP}:${PORT}/"; SURL="https://${IP}:${PORT}/"

open_box() {   # where to open it, once it answers
  section "Open Neat Change Studio"
  kv "Web" "$(link "$URL")"
  ((HTTPS)) && kv "Webcam" "$(link "$SURL") ${C_MUTED}(same port; accept the browser's warning once)${C_RESET}"
  ((daemon)) && kv "Log" "$HERE/server.log"
  kv "Stop" "neat-change --stop"
  printf '\n'
}

if status_json | grep -q '"cyws-demo"'; then
  banner
  ok "Already running at $(link "$URL")"
  printf '\n'; exit 0
fi

MODELS_DIR="$(cyws_models_dir "$ROOT")"
banner "${C_MUTED}Model${C_RESET}   $(title_of "$MODEL")" "${C_MUTED}Board${C_RESET}   $(hostname 2>/dev/null || echo board) · ${IP}" \
       "${C_MUTED}Port${C_RESET}    ${PORT}$( ((HTTPS)) && echo " · HTTP and HTTPS")"

section "System"
pyver="$("$VENV/bin/python3" -c 'import importlib.metadata as m; print(m.version("pyneat"))' 2>/dev/null || true)"
kv "pyneat" "${pyver:-not found in $VENV}"
kv "python" "$("$VENV/bin/python3" -c 'import platform; print(platform.python_version())' 2>/dev/null || echo unknown)"
have=()
for f in "$MODELS_DIR"/*_mpk.tar.gz; do
  [[ -e "$f" ]] || continue
  case "$(basename "$f")" in
    cyws_bf16_mpk.tar.gz) have+=("CYWS bf16") ;; cyws_w8a16_mpk.tar.gz) have+=("CYWS w8a16") ;;
    dinov2_vits14_reg_448_bf16_mpk.tar.gz) have+=("DINOv2 448") ;; *) have+=("$(basename "$f" _mpk.tar.gz)") ;;
  esac
done
kv "models" "$( ((${#have[@]})) && { IFS='|'; printf '%s' "${have[*]}" | sed 's/|/ · /g'; } || echo "none yet (download one in Models, or: neat-change --fetch bf16)")"
kv "folder" "$MODELS_DIR"

step "Checking the MLA"
# One MLA user at a time (matches full command lines of other demos, not this script).
# The pattern lists command-line names of other MLA apps that may be running
# (bracketed first letters keep pgrep from matching itself); set
# MLA_BUSY_PATTERN to match your own. setup.sh --check uses the same default; keep the two
# alike. A demo from this install on this port was found above; [w]ebapp/server.py here is
# one on another port.
MLA_BUSY_PATTERN="${MLA_BUSY_PATTERN:-[n]eat-genai-studio|[l]aya/|[c]ar-scratch|[d]ofa|[l]lima|[s]rc/app.py --config|[m]la_client|[w]ebapp/server.py|[r]un_cyws_neat.py}"
busy="$(pgrep -af "$MLA_BUSY_PATTERN" || true)"
if [[ -n "$busy" && -z "${FORCE:-}" ]]; then
  errln "Another app may be using the MLA:"
  while IFS= read -r line; do printf '      %s%s%s\n' "${C_MUTED}" "${line:0:110}" "${C_RESET}" >&2; done <<< "$busy"
  info "Stop it first, or start with FORCE=1 (only one process may use the MLA)" >&2
  exit 1
fi
[[ -n "$busy" ]] && warn "Another app may be using the MLA (FORCE=1: starting anyway)" || ok "Free: no other MLA app is running"
[[ -f "$VENV/bin/activate" ]] || { errln "pyneat environment not found at $VENV (set PYNEAT_VENV)"; exit 1; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$ROOT"
export PYTHONUNBUFFERED=1
if ((HTTPS)); then
  step "Certificate"
  if bash "$HERE/make_cert.sh" "$HERE/certs" > /dev/null 2>&1; then
    ok "HTTPS on port ${PORT} too, for the webcam (self-signed: the browser warns once)"
  else warn "No certificate (is openssl installed?): plain HTTP only, so the webcam works only on the board itself"; fi
fi

# Follows the server until its model is ready: one line per load stage (a spinner in --daemon
# mode on a terminal). Returns 0 once it answers (ready, or idle with no model), 1 on an error.
follow_load() {   # follow_load <server pid>
  local spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏' i=0 t0=$SECONDS shown="" line state stage title err msg
  for _ in $(seq 1 600); do
    if ! kill -0 "$1" 2>/dev/null; then
      ((daemon && TERM_FANCY)) && printf '\r\033[K'
      errln "The server stopped while starting."; ((daemon)) && tail -n 5 "$HERE/server.log" | sed 's/^/      /' >&2
      return 1
    fi
    if line="$(status_json | status_line)"; then
      IFS=$'\x1f' read -r state stage title err <<< "$line"   # not tabs: read would merge empty fields
      case "$state:$stage" in
        ready:*) ((daemon && TERM_FANCY)) && printf '\r\033[K'
                 ok "${C_BOLD}${title}${C_RESET} is ready on the MLA ($((SECONDS - t0)) s)"; return 0 ;;
        idle:*)  ((daemon && TERM_FANCY)) && printf '\r\033[K'
                 warn "Running with no model loaded: ${err:-none is on this board}"
                 info "Download one in the Models window, or: neat-change --fetch bf16"; return 0 ;;
        error:*) ((daemon && TERM_FANCY)) && printf '\r\033[K'
                 errln "${err:-The model did not load}"; return 1 ;;
        *:release) msg="Releasing the MLA" ;;
        *:warm) msg="Warming up ${title}" ;;
        *) msg="Loading ${title:-$(title_of "$MODEL")} on the MLA" ;;
      esac
    else msg="Starting the server"; fi
    if ((daemon && TERM_FANCY)); then
      printf '\r\033[K  %s%s%s %s %s%d s%s' "${C_ACCENT}" "${spin:i++%10:1}" "${C_RESET}" "$msg…" "${C_MUTED}" $((SECONDS - t0)) "${C_RESET}"
    elif [[ "$msg" != "$shown" ]]; then
      info "$msg…"; shown="$msg"
    fi
    sleep 0.2
  done
  ((daemon && TERM_FANCY)) && printf '\r\033[K'
  warn "Still not ready after two minutes; see $HERE/server.log"; return 1
}

step "Starting Neat Change Studio"
if ((daemon)); then
  setsid nohup python3 "$HERE/server.py" "${args[@]}" > "$HERE/server.log" 2>&1 < /dev/null &
  if follow_load $!; then open_box; else exit 1; fi
  exit 0
fi
if ((TERM_FANCY)); then
  # In front, on a terminal: the server's log follows the steps, and the links are shown once it
  # answers. A background job of a script ignores Ctrl+C, so Ctrl+C (and TERM, HUP) is passed on
  # as a TERM, and the script waits until the server has let go of the MLA.
  python3 "$HERE/server.py" "${args[@]}" &
  server=$!
  trap 'kill -TERM "$server" 2>/dev/null' INT TERM HUP
  ( follow_load "$server" && open_box ) &
  watcher=$!
  set +e
  wait "$server"; rc=$?
  while kill -0 "$server" 2>/dev/null; do wait "$server"; rc=$?; done   # a trapped signal ends the first wait early
  set -e
  kill "$watcher" 2>/dev/null || true
  exit "$rc"
fi
exec python3 "$HERE/server.py" "${args[@]}"
