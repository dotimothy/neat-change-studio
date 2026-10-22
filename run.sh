#!/usr/bin/env bash
# Run the CYWS change-detection CLI on the DevKit. All arguments go to
# app/run_cyws_neat.py (see --help):
#   ./run.sh --mode static --before b.jpg --after a.jpg
#   ./run.sh --mode dataset --dataset-dir data/testCYWS --levels 0,1 --tiles gated --bench
#
# Defaults added when not given: --model-path models/cyws_bf16_mpk.tar.gz (or the
# only MPK in models/) for the neat backend, and --out ./out. Relative paths are
# resolved against the directory you call this from, not app/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
VENV="${PYNEAT_VENV:-$HOME/pyneat}"

if [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
else
  echo "note: pyneat environment not found at $VENV; using $(command -v python3)" >&2
fi

# Make path options absolute (we cd into app/ below) and note what was given.
args=()
backend="" has_model=0 has_out=0
path_opts=" --before --after --mask --dataset-dir --model-path --out "
abspath() { [[ "$1" == /* || "$1" == "-" ]] && printf '%s' "$1" || printf '%s/%s' "$PWD" "$1"; }
while (($#)); do
  a="$1"; shift
  case "$a" in
    --*=*)
      opt="${a%%=*}" val="${a#*=}"
      [[ "$path_opts" == *" $opt "* ]] && val="$(abspath "$val")"
      [[ "$opt" == --backend ]] && backend="$val"
      [[ "$opt" == --model-path ]] && has_model=1
      [[ "$opt" == --out ]] && has_out=1
      args+=("$opt=$val") ;;
    --*)
      args+=("$a")
      if [[ "$path_opts" == *" $a "* || "$a" == --backend ]] && (($#)); then
        val="$1"; shift
        [[ "$a" == --backend ]] && backend="$val" || val="$(abspath "$val")"
        [[ "$a" == --model-path ]] && has_model=1
        [[ "$a" == --out ]] && has_out=1
        args+=("$val")
      fi ;;
    *) args+=("$a") ;;
  esac
done

if ((!has_model)) && [[ -z "$backend" || "$backend" == neat ]]; then
  mpk="$HERE/models/cyws_bf16_mpk.tar.gz"
  if [[ ! -f "$mpk" ]]; then
    shopt -s nullglob; found=("$HERE"/models/*_mpk.tar.gz); shopt -u nullglob
    ((${#found[@]} == 1)) && mpk="${found[0]}" || mpk=""
  fi
  if [[ -n "$mpk" ]]; then
    args+=(--model-path "$mpk")
  else
    echo "No model in $HERE/models. Download one from Hugging Face first:" >&2
    echo "  $HERE/webapp/run.sh --fetch bf16      (or: w8a16, all)" >&2
    exit 1
  fi
fi
((has_out)) || args+=(--out "$HERE/out")

# One MLA user at a time: a second runner makes both fail or hang.
# pgrep -f matches whole command lines, so this script, the shells that
# started it (ssh "cd ... && ./run.sh ...") and its own subshells would match
# on user arguments such as ~/photos/before.jpg; skip that process tree.
# The pattern lists command-line names of other MLA apps that may be running
# (bracketed first letters keep pgrep from matching itself); set
# MLA_BUSY_PATTERN to match your own.
mla_users() {
  local pattern="${MLA_BUSY_PATTERN:-[n]eat-genai-studio|[l]aya|[c]ar-scratch|[d]ofa|[l]lima}"
  local own=" $$ $BASHPID " skip=" $$ $BASHPID " p=$$ pid rest ppid
  while p="$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')" && [[ -n "$p" && "$p" != 0 ]]; do
    skip+="$p "                                                           # our ancestors
  done
  while read -r pid rest; do
    [[ -z "$pid" || "$skip" == *" $pid "* ]] && continue
    ppid="$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')"
    [[ -z "$ppid" || "$own" == *" $ppid "* ]] && continue   # gone, or one of our subshells
    printf '%s %s\n' "$pid" "$rest"
  done <<<"$(pgrep -af "$pattern" || true)"
}
if [[ -z "$backend" || "$backend" == neat ]] && [[ -z "${FORCE:-}" ]]; then
  busy="$(mla_users)"
  if [[ -n "$busy" ]]; then
    echo "Another app may be using the MLA:" >&2; echo "$busy" >&2
    echo "Stop it first, or run with FORCE=1 ./run.sh ..." >&2
    exit 1
  fi
fi

cd "$HERE/app"
export PYTHONUNBUFFERED=1
exec python3 run_cyws_neat.py "${args[@]}"
