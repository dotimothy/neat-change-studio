#!/usr/bin/env bash
# Set up Neat Change Studio on the Modalix DevKit: check the pyneat venv and install
# the Python packages it is missing, download a model from Hugging Face, make the
# HTTPS certificate for the web demo, and add the 'neat-change' alias and a desktop
# icon. Safe to run again. ./setup.sh --help lists the options.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
VENV="${PYNEAT_VENV:-$HOME/pyneat}"
# shellcheck source=webapp/models_dir.sh
source "$HERE/webapp/models_dir.sh"
MODELS_DIR="$(cyws_models_dir "$HERE")"
PORT=8098
ALIAS_NAME=neat-change
APP_NAME="Neat Change Studio"

# ---------------------------------------------------------------------------
# Output: the Neat sparkle banner and coloured status lines. Plain text when
# stdout is not a terminal, TERM is "dumb", or NO_COLOR is set.
# ---------------------------------------------------------------------------
if [[ -t 1 && -z "${NO_COLOR:-}" && "${TERM:-}" != "dumb" ]]; then
  C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'
  P_TEAL=$'\033[38;2;61;179;138m'
  P_GREEN=$'\033[38;2;74;168;54m'
  P_LIME=$'\033[38;2;154;190;30m'
  P_BLUE=$'\033[38;2;58;125;216m'
  P_ORANGE=$'\033[38;2;223;108;30m'
  P_INK=$'\033[38;2;60;66;74m'
  C_ACCENT="${P_LIME}"; C_MUTED=$'\033[38;2;140;150;160m'
  C_OK=$'\033[38;2;53;196;137m'; C_WARN=$'\033[38;2;224;173;74m'; C_ERR=$'\033[38;2;239;91;98m'
else
  C_RESET=''; C_BOLD=''; C_DIM=''
  P_TEAL=''; P_GREEN=''; P_LIME=''; P_BLUE=''; P_ORANGE=''; P_INK=''
  C_ACCENT=''; C_MUTED=''; C_OK=''; C_WARN=''; C_ERR=''
fi

step() { printf '\n%s\n' "${C_ACCENT}${C_BOLD}▸${C_RESET} $*"; }
info() { printf '%s\n' "${C_MUTED}·${C_RESET} $*"; }
ok()   { printf '%s\n' "${C_OK}✔${C_RESET} $*"; }
warn() { printf '%s\n' "${C_WARN}⚠${C_RESET} $*" >&2; }
errln(){ printf '%s\n' "${C_ERR}✘${C_RESET} $*" >&2; }

banner() {
  printf '\n'
  printf '        %s▲%s\n'                   "${P_TEAL}"   "${C_RESET}"
  printf '       %s███%s\n'                  "${P_GREEN}"  "${C_RESET}"
  printf '      %s█████%s\n'                 "${P_GREEN}"  "${C_RESET}"
  printf '   %s◀████%s█%s████▶%s\n'          "${P_BLUE}" "${P_INK}" "${P_LIME}" "${C_RESET}"
  printf '      %s█████%s\n'                 "${P_ORANGE}" "${C_RESET}"
  printf '       %s███%s\n'                  "${P_ORANGE}" "${C_RESET}"
  printf '        %s▼%s\n'                   "${P_ORANGE}" "${C_RESET}"
  printf '\n'
  printf '   %sNEAT%s %sChange Studio%s  %s· setup%s\n' \
    "${C_BOLD}" "${C_RESET}" "${C_ACCENT}${C_BOLD}" "${C_RESET}" "${C_MUTED}" "${C_RESET}"
  printf '   %sChecks the pyneat environment, fetches a model and sets up the web demo%s\n' "${C_MUTED}" "${C_RESET}"
  printf '\n'
}

# A divider of Neat sparkles (✦) in the logo colours, then a bold heading.
section() {
  local palette=("${P_TEAL}" "${P_GREEN}" "${P_LIME}" "${P_BLUE}" "${P_ORANGE}")
  local line="   " i
  for ((i = 0; i < 12; i++)); do line+="${palette[i % 5]}✦${C_RESET} "; done
  printf '\n%s\n   %s%s%s\n' "${line}" "${C_BOLD}" "$1" "${C_RESET}"
}

usage() {
  cat <<'USAGE'
Usage:
  ./setup.sh [options]

Sets up Neat Change Studio on the Modalix DevKit, in order:
  1. checks the pyneat venv and installs the Python packages it is missing
     (numpy, opencv-python-headless with SIFT, pillow, flask)
  2. downloads a model from Hugging Face (dotimothy/sima-cyws) into the models
     directory (default /media/nvme/models/cyws, shared by every install)
  3. makes the self-signed certificate the web demo serves HTTPS with
  4. adds the 'neat-change' shell alias for webapp/run.sh, and a desktop icon
     when the board has a desktop
Safe to run again: what is already in place is kept.

Options:
  --check-only          Only check the environment; install, download and
                        change nothing. Exits 1 when a Python package is
                        missing (a missing model is only reported).
  --check               At the end, load the model on the MLA and run one
                        synthetic pair (refuses while another MLA app runs)
  --models LIST         Models to download: bf16, w8a16, all or none
                        (same as CYWS_MODELS; LIST may be bf16,w8a16)
  --fetch[=LIST]        Same as --models; plain --fetch means bf16
  --models-dir DIR      Keep the models in DIR, and remember it (in .models-dir)
                        for run.sh, the web demo and the alias. 'default' goes
                        back to /media/nvme/models/cyws. Or CYWS_MODELS_DIR=DIR.
  --no-alias            Do not add the shell alias      (CREATE_ALIAS=0)
  --no-desktop-icon     Do not add the desktop icon     (CREATE_DESKTOP_ICON=0)
  --no-prompt           Never ask; decide as without a terminal
  --install             Accepted for older scripts; installing missing
                        packages is the default now
  -h, --help            Show this help

Environment:
  PYNEAT_VENV           The pyneat venv. Default: ~/pyneat
  CYWS_MODELS_DIR       The models directory for this run (see --models-dir).
  CYWS_MODELS           bf16 (119 MB, most accurate), w8a16 (72 MB, faster),
                        all or none. If unset: in a terminal, setup asks
                        (default bf16) when there is no model yet; without
                        a terminal it downloads none.
  CREATE_ALIAS          1 or 0: add 'neat-change' to the shell's startup files.
                        If unset: in a terminal, setup asks (default yes);
                        otherwise 0.
  CREATE_DESKTOP_ICON   1 or 0: add a desktop icon and menu entry that start
                        the demo and open it in the browser. If unset: in a
                        terminal, setup asks (default yes) when the board has a
                        desktop; otherwise 0, but an icon already made for
                        this install is kept up to date.
  MLA_BUSY_PATTERN      Command lines --check (and webapp/run.sh) treat as
                        other MLA apps.
  NO_COLOR              Plain output without colours.

Prompts are read from the terminal (/dev/tty), so they also work under
'curl ... | bash' with STUDIO_SETUP=1.
USAGE
}

CHECK_ONLY=0 CHECK=0 NO_PROMPT=0
MODELS="${CYWS_MODELS-}"
MODELS_SET=0; [[ -n "${CYWS_MODELS+x}" ]] && MODELS_SET=1
WANT_ALIAS="${CREATE_ALIAS:-}"
WANT_ICON="${CREATE_DESKTOP_ICON:-}"
while (($#)); do
  case "$1" in
    --check-only) CHECK_ONLY=1 ;;
    --check) CHECK=1 ;;
    --models) [[ $# -ge 2 ]] || { echo "--models needs a value" >&2; exit 2; }
              MODELS="$2"; MODELS_SET=1; shift ;;
    --models=*) MODELS="${1#--models=}"; MODELS_SET=1 ;;
    --fetch) MODELS=bf16; MODELS_SET=1 ;;
    --fetch=*) MODELS="${1#--fetch=}"; MODELS_SET=1 ;;
    --models-dir|--models-dir=*)
      if [[ "$1" == *=* ]]; then dir="${1#--models-dir=}"; else
        [[ $# -ge 2 ]] || { echo "--models-dir needs a directory (or 'default')" >&2; exit 2; }; dir="$2"; shift; fi
      if [[ "$dir" == default ]]; then rm -f "$HERE/.models-dir"; unset CYWS_MODELS_DIR
      else [[ "$dir" == "~"* ]] && dir="$HOME${dir:1}"; dir="$(mkdir -p "$dir" && cd "$dir" && pwd -P)" \
             || { echo "cannot use $dir as the models directory" >&2; exit 2; }
           printf '%s\n' "$dir" > "$HERE/.models-dir"; export CYWS_MODELS_DIR="$dir"; fi
      MODELS_DIR="$(cyws_models_dir "$HERE")" ;;
    --no-alias) WANT_ALIAS=0 ;;
    --no-desktop-icon) WANT_ICON=0 ;;
    --no-prompt) NO_PROMPT=1 ;;
    --install) ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1 (see ./setup.sh --help)" >&2; exit 2 ;;
  esac
  shift
done

# Ask on the terminal even when stdin is a pipe (curl | bash). Without one, or with
# --no-prompt, ask() is never called and the defaults for no terminal apply.
TTY=0
if ((!NO_PROMPT)) && { : < /dev/tty; } 2> /dev/null; then TTY=1; fi
ask() {   # ask "question" default → the answer, or the default on Enter / EOF
  local answer=""
  printf '%s ' "$1" > /dev/tty
  read -r answer < /dev/tty || answer=""
  printf '%s' "${answer:-$2}"
}
yes_no() {   # yes_no "question" y|n → 0 for yes
  local hint="[y/N]" a; [[ "$2" == y ]] && hint="[Y/n]"
  a="$(ask "$1 $hint" "$2")"
  [[ "${a,,}" == y || "${a,,}" == yes ]]
}

FAILED=()
fail_step() { errln "$1"; FAILED+=("$1"); }

banner
info "Install: ${C_DIM}${HERE}${C_RESET}"

# --------------------------------------------------------------------------- python
section "Python Environment"
step "pyneat venv"
if [[ ! -f "$VENV/bin/activate" ]]; then
  errln "pyneat environment not found at $VENV"
  info "It comes with the NEAT SDK on the DevKit; set PYNEAT_VENV if it is somewhere else."
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
ok "$VENV ($(python3 -c 'import sys; print(sys.version.split()[0])'))"

MISSING_FILE="$(mktemp)"
CHECK_DIR=""
trap 'rm -f "$MISSING_FILE"; [[ -n "$CHECK_DIR" ]] && rm -rf "$CHECK_DIR"; true' EXIT
export MISSING_FILE
check_packages() {
  python3 -W ignore - <<'PY'
import importlib, os, sys
missing = []
for mod, pkg in (("numpy", "numpy"), ("cv2", "opencv-python-headless"), ("PIL", "pillow"), ("flask", "flask")):
    try:
        m = importlib.import_module(mod)
        print(f"ok   {mod} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"miss {mod} ({e})")
        missing.append(pkg)
if "opencv-python-headless" not in missing:
    import cv2
    if hasattr(cv2, "SIFT_create"):
        cv2.SIFT_create(100)
        print("ok   cv2.SIFT_create")
    else:
        print("warn cv2.SIFT_create is missing: photo alignment falls back to ORB "
              "(pip install -U 'opencv-python-headless>=4.4' adds it)")
try:
    import scipy
    print(f"ok   scipy {scipy.__version__} (optional)")
except Exception:
    print("info scipy is not installed (optional)")
with open(os.environ["MISSING_FILE"], "w") as f:
    f.write(" ".join(missing))
sys.exit(1 if missing else 0)
PY
}
show_packages() {   # turn check_packages' lines into status lines
  local kind rest
  while IFS= read -r line; do
    kind="${line%% *}"; rest="${line#* }"; rest="${rest#"${rest%%[! ]*}"}"
    case "$kind" in
      ok) ok "$rest" ;; miss) warn "missing: $rest" ;; warn) warn "$rest" ;; *) info "$rest" ;;
    esac
  done
}

step "Python packages"
if out="$(check_packages)"; then
  show_packages <<< "$out"
else
  show_packages <<< "$out"
  missing="$(cat "$MISSING_FILE")"
  if ((CHECK_ONLY)); then
    fail_step "Missing packages: $missing (./setup.sh installs them)"
  else
    info "Installing into the pyneat venv: ${C_BOLD}${missing}${C_RESET}"
    # shellcheck disable=SC2086
    if python3 -m pip install --quiet $missing && out="$(check_packages)"; then
      show_packages <<< "$out"
      ok "Packages installed."
    else
      errln "Could not install: $missing"
      info "Try by hand: $VENV/bin/python3 -m pip install $missing"
      exit 1
    fi
  fi
fi

step "pyneat"
if info_line="$(python3 -c 'import pyneat
print(pyneat.build_info() if hasattr(pyneat, "build_info") else "(build_info() not available)")' 2>&1)"; then
  ok "pyneat: $(tail -n 1 <<< "$info_line")"
else
  errln "pyneat is not importable in $VENV:"
  printf '%s\n' "$info_line" | tail -n 3 >&2
  exit 1
fi

# --------------------------------------------------------------------------- models
section "Model"
shopt -s nullglob
info "Models directory: ${C_DIM}$MODELS_DIR${C_RESET}"
((CHECK_ONLY)) || mkdir -p "$MODELS_DIR" 2>/dev/null || warn "Cannot create $MODELS_DIR (set another with --models-dir DIR)."
# Models left in this app's own models/ from before the shared directory: name them and the move.
if [[ "$(cd "$HERE" && pwd -P)/models" != "$(cd "$MODELS_DIR" 2>/dev/null && pwd -P || echo "$MODELS_DIR")" ]]; then
  old=("$HERE"/models/*_mpk.tar.gz)
  if ((${#old[@]})); then
    warn "${#old[@]} model(s) in $HERE/models are not used any more. Move them with:"
    info "  mv $(printf '%q ' "${old[@]}")$(printf '%q' "$MODELS_DIR")/"
  fi
fi
have_models() { local m=("$MODELS_DIR"/*_mpk.tar.gz); ((${#m[@]})); }
list_models() { local m; for m in "$MODELS_DIR"/*_mpk.tar.gz; do ok "Model: ${C_DIM}$m${C_RESET} ($(du -h "$m" | cut -f1))"; done; }

if ((CHECK_ONLY)); then
  if have_models; then list_models; else info "No model in $MODELS_DIR yet."; fi
else
  if ((!MODELS_SET)); then
    if have_models; then
      MODELS=none
    elif ((TTY)); then
      info "The compiled models are on Hugging Face (dotimothy/sima-cyws):"
      info "  ${C_BOLD}bf16${C_RESET}   119 MB, the most accurate (default)"
      info "  ${C_BOLD}w8a16${C_RESET}   72 MB, faster"
      MODELS="$(ask "Download which model? [bf16/w8a16/all/none] (bf16)" bf16)"
    else
      MODELS=none
      info "No terminal to ask on and CYWS_MODELS is not set: no model downloaded."
    fi
  fi
  MODELS="${MODELS,,}"; MODELS="${MODELS//,/ }"
  bad=""
  for m in $MODELS; do [[ "$m" =~ ^(bf16|w8a16|all|none)$ ]] || bad+=" $m"; done
  if [[ -n "$bad" ]]; then
    fail_step "Unknown model:${bad} (choose bf16, w8a16, all or none)"
  elif [[ -n "${MODELS// /}" && " $MODELS " != *" none "* ]]; then
    step "Downloading from Hugging Face: $MODELS"
    # shellcheck disable=SC2086
    if "$HERE/webapp/run.sh" --models "$MODELS_DIR" --fetch $MODELS; then
      ok "Download done."
    else
      fail_step "A model download did not finish; run ./setup.sh again to continue it."
    fi
  fi
  if have_models; then
    list_models
  else
    info "No model in $MODELS_DIR yet. Download one with ${C_BOLD}CYWS_MODELS=bf16 ./setup.sh${C_RESET},"
    info "${C_BOLD}webapp/run.sh --fetch bf16${C_RESET}, or in the web demo under Settings → Model."
  fi
fi
[[ -d "$HERE/data/testCYWS" ]] && info "Test pairs: $HERE/data/testCYWS" \
  || info "No test pairs in data/testCYWS (optional; see the README, \"Dataset\")."

# --------------------------------------------------------------------------- web demo
section "Web Demo"
step "HTTPS certificate"
CERTS="$HERE/webapp/certs"
if [[ -f "$CERTS/cert.pem" && -f "$CERTS/key.pem" ]]; then
  ok "Certificate: ${C_DIM}$CERTS/cert.pem${C_RESET}"
elif ((CHECK_ONLY)); then
  info "No certificate yet; webapp/run.sh makes one on first start."
elif msg="$(bash "$HERE/webapp/make_cert.sh" "$CERTS" 2>&1)"; then
  ok "${msg^}"
  info "Browsers warn about a self-signed certificate once (Advanced → continue); HTTPS is what lets the webcam work."
else
  warn "${msg:-could not make a certificate}; the demo serves plain HTTP until there is one."
fi

# --------------------------------------------------------------------------- shortcuts
# The alias goes where this user's bash looks: ~/.bashrc for terminals on the desktop, and
# the login file (the first that exists of ~/.bash_profile, ~/.bash_login, ~/.profile) for
# ssh; ~/.zshrc for zsh.
alias_files() {
  local file login=""
  [[ -f "$HOME/.bashrc" ]] && printf '%s\n' "$HOME/.bashrc"
  for file in "$HOME/.bash_profile" "$HOME/.bash_login" "$HOME/.profile"; do
    [[ -f "$file" ]] && { login="$file"; break; }
  done
  printf '%s\n' "${login:-$HOME/.bash_profile}"
  [[ "$(basename "${SHELL:-bash}")" == zsh ]] && printf '%s\n' "$HOME/.zshrc"
  return 0
}
# alias neat-change='<install>/webapp/run.sh', with any ' in the path escaped.
alias_line() {
  local path="${HERE}/webapp/run.sh" q="'\\''"
  printf "alias %s='%s'" "$ALIAS_NAME" "${path//\'/$q}"
}
install_alias() {
  local line file here=1 want="$WANT_ALIAS"
  line="$(alias_line)"
  local -a files
  mapfile -t files < <(alias_files)
  for file in "${files[@]}"; do
    [[ -f "$file" ]] && grep -qxF "$line" "$file" || here=0
  done
  if ((here)) && [[ "$want" != 0 ]]; then
    ok "'${ALIAS_NAME}' alias already set in ${C_DIM}${files[*]}${C_RESET}"
    return 0
  fi
  if [[ -z "$want" ]]; then
    if ((TTY)); then
      yes_no "Add a '${ALIAS_NAME}' shell alias (starts the web demo)?" y && want=1 || want=0
    else
      want=0
    fi
  fi
  if [[ "$want" != 1 ]]; then
    info "Shell alias not added (CREATE_ALIAS=1 adds it). Start the demo with ${C_BOLD}$(printf '%q' "${HERE}/webapp/run.sh")${C_RESET}."
    return 0
  fi
  for file in "${files[@]}"; do
    if [[ -f "$file" ]] && grep -qxF "$line" "$file"; then
      ok "'${ALIAS_NAME}' alias already in ${C_DIM}${file}${C_RESET}"
    elif [[ -f "$file" ]] && grep -q "^alias ${ALIAS_NAME}=" "$file"; then
      # Set up before from another directory. awk takes the line from the environment, so
      # no character in the path is special; cat keeps the file's owner, mode and links.
      local tmp
      tmp="$(mktemp)" && LINE="$line" awk -v pre="alias ${ALIAS_NAME}=" \
        'index($0, pre) == 1 { print ENVIRON["LINE"]; next } { print }' "$file" > "$tmp" \
        && cat "$tmp" > "$file"
      rm -f "$tmp"
      ok "'${ALIAS_NAME}' alias updated in ${C_DIM}${file}${C_RESET}"
    else
      printf '\n# %s, added by setup.sh\n%s\n' "$APP_NAME" "$line" >> "$file"
      ok "Added a '${ALIAS_NAME}' alias to ${C_DIM}${file}${C_RESET}"
    fi
  done
  ALIAS_ADDED=1
  info "It works in a new shell, or in this one after: ${C_BOLD}source $(printf '%q' "${files[0]}")${C_RESET}"
}

# A launcher on the desktop and in the applications menu: webapp/desktop-launch.sh starts
# the demo in the background (or finds it running) and opens it in the board's browser.
# Exec= quoting follows the desktop entry spec ("..." with \ before " ` $ \, and %%). A terminal's
# -e is not portable for a path with spaces or quotes (xfce4-terminal splits it again), so
# such a path uses Terminal=true instead of x-terminal-emulator.
desktop_entry() {
  local launch="$HERE/webapp/desktop-launch.sh" exec_line terminal=true quoted
  quoted="${launch//\\/\\\\}"; quoted="${quoted//\"/\\\"}"; quoted="${quoted//\`/\\\`}"; quoted="${quoted//\$/\\\$}"; quoted="${quoted//%/%%}"
  exec_line="Exec=\"${quoted}\""
  if [[ "$launch" =~ ^[A-Za-z0-9_./+-]+$ ]] && command -v x-terminal-emulator > /dev/null 2>&1; then
    exec_line="Exec=x-terminal-emulator -T \"${APP_NAME}\" -e \"${launch}\""
    terminal=false
  fi
  printf '%s\n' "[Desktop Entry]" "Version=1.0" "Type=Application" "Name=${APP_NAME}" \
    "Comment=Before/after change detection with CYWS on the Modalix MLA" \
    "$exec_line" "Path=${HERE}" "Icon=${HERE}/webapp/web/assets/neat-mark.png" \
    "Terminal=${terminal}" "Categories=Development;" "Keywords=CYWS;Change;SiMa;Modalix;Neat;" \
    "StartupNotify=false"
}
install_icon() {
  local apps_dir="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
  local desktop_dir="${XDG_DESKTOP_DIR:-$HOME/Desktop}" want="$WANT_ICON"
  local installed="$apps_dir/neat-change-studio.desktop"
  if [[ ! -d "$desktop_dir" && ! -d /usr/share/xsessions ]]; then
    [[ "$want" == 1 ]] && info "This board has no desktop; no icon installed."
    return 0
  fi
  if [[ -z "$want" && -f "$installed" ]] && grep -qxF "Path=${HERE}" "$installed"; then
    want=1                                   # keep this install's icon up to date
  fi
  if [[ -z "$want" ]]; then
    if ((TTY)); then
      yes_no "Add a desktop icon that starts the demo and opens it in the browser?" y && want=1 || want=0
    else
      want=0
    fi
  fi
  if [[ "$want" != 1 ]]; then
    info "Desktop icon not added (CREATE_DESKTOP_ICON=1 adds it)."
    return 0
  fi
  chmod +x "$HERE/webapp/desktop-launch.sh" "$HERE/webapp/run.sh" 2> /dev/null || true
  local entry target trusted=1 bus="${DBUS_SESSION_BUS_ADDRESS:-}"
  local -a targets=("$installed")
  [[ -d "$desktop_dir" ]] && targets+=("$desktop_dir/neat-change-studio.desktop")
  [[ -z "$bus" && -S "/run/user/$(id -u)/bus" ]] && bus="unix:path=/run/user/$(id -u)/bus"
  entry="$(desktop_entry)"
  mkdir -p "$apps_dir"
  for target in "${targets[@]}"; do
    printf '%s\n' "$entry" > "$target"
    chmod 755 "$target"
    # XFCE asks before running a launcher it does not trust. It can be told through the
    # desktop session's bus, when somebody is logged in at the board.
    if [[ -n "$bus" ]] && command -v gio > /dev/null 2>&1 \
       && DBUS_SESSION_BUS_ADDRESS="$bus" gio set -t string "$target" metadata::xfce-exe-checksum \
            "$(sha256sum "$target" | cut -d' ' -f1)" > /dev/null 2>&1; then
      DBUS_SESSION_BUS_ADDRESS="$bus" gio set -t string "$target" metadata::trusted true > /dev/null 2>&1 || true
    else
      trusted=0
    fi
  done
  command -v update-desktop-database > /dev/null 2>&1 && update-desktop-database "$apps_dir" > /dev/null 2>&1 || true
  ok "Desktop icon: ${C_DIM}${targets[*]}${C_RESET}"
  info "Double-click ${C_BOLD}${APP_NAME}${C_RESET} on the desktop or in the applications menu: the demo starts and the browser opens on it."
  ((trusted)) || info "The first double-click may ask to trust the launcher; choose ${C_BOLD}Mark Executable${C_RESET}."
}

ALIAS_ADDED=0
if ((!CHECK_ONLY)); then
  section "Shortcuts"
  install_alias
  install_icon
fi

# --------------------------------------------------------------------------- MLA check
# One synthetic pair through ./run.sh: the model loads on the MLA and runs. Only one
# process may use the MLA, so this refuses while another MLA app (or a CYWS demo) runs.
mla_check() {
  # The same default as webapp/run.sh; keep the two alike.
  local pattern="${MLA_BUSY_PATTERN:-[n]eat-genai-studio|[l]aya/|[c]ar-scratch|[d]ofa|[l]lima|[s]rc/app.py --config|[m]la_client|[w]ebapp/server.py|[r]un_cyws_neat.py}"
  local busy
  busy="$(pgrep -af "$pattern" || true)"
  if [[ -n "$busy" ]]; then
    fail_step "Another app may be using the MLA; stop it, then run ./setup.sh --check again:"
    printf '%s\n' "$busy" >&2
    return 0
  fi
  if ! have_models; then
    fail_step "No model in $MODELS_DIR to check; download one first (CYWS_MODELS=bf16 ./setup.sh)."
    return 0
  fi
  CHECK_DIR="$(mktemp -d)" || { fail_step "MLA check: mktemp failed."; return 0; }
  python3 - "$CHECK_DIR" <<'PY' || { fail_step "MLA check: could not make the synthetic pair."; return 0; }
import sys
import numpy as np
from PIL import Image
rng = np.random.default_rng(0)
before = (rng.random((96, 128, 3)) * 255).astype(np.uint8)
before = np.array(Image.fromarray(before).resize((512, 384), Image.BILINEAR))
after = before.copy()
after[140:240, 200:330] = (30, 200, 60)                 # something appeared
Image.fromarray(before).save(f"{sys.argv[1]}/check_before.jpg", quality=92)
Image.fromarray(after).save(f"{sys.argv[1]}/check_after.jpg", quality=92)
PY
  info "Running one synthetic pair on the MLA…"
  local log="$CHECK_DIR/run.log" start end
  start=$(date +%s)
  if "$HERE/run.sh" --mode static --before "$CHECK_DIR/check_before.jpg" --after "$CHECK_DIR/check_after.jpg" \
       --preset fast --out "$CHECK_DIR/out" > "$log" 2>&1; then
    end=$(date +%s)
    local found
    found="$(python3 - "$CHECK_DIR/out" <<'PY'
import json, sys
from pathlib import Path
for p in sorted(Path(sys.argv[1]).glob("*.json")):
    if p.name == "metrics.json":
        continue
    try:
        d = json.loads(p.read_text())
        print(len(d.get("detections", [])))
    except Exception:
        print("?")
    break
else:
    print("?")
PY
)" || found="?"
    ok "MLA check passed: the model loaded and ran ($found detection(s) on the synthetic pair, $((end - start)) s in all)."
  else
    tail -n 15 "$log" >&2
    fail_step "MLA check failed (the log is above)."
  fi
}
if ((CHECK)); then
  section "MLA Check"
  mla_check
fi

# --------------------------------------------------------------------------- summary
section "Done"
if ((${#FAILED[@]})); then
  for f in "${FAILED[@]}"; do errln "$f"; done
  printf '\n'
  exit 1
fi
if ((CHECK_ONLY)); then
  ok "The environment is ready."
  printf '\n'
  exit 0
fi
ok "Setup complete."
ip="$(hostname -I 2> /dev/null | awk '{print $1}' || true)"; ip="${ip:-<board-ip>}"
run="$(printf '%q' "$HERE/webapp/run.sh")"
if ((ALIAS_ADDED)); then
  run="$ALIAS_NAME"
else
  while IFS= read -r f; do
    [[ -f "$f" ]] && grep -qxF "$(alias_line)" "$f" && { run="$ALIAS_NAME"; break; }
  done < <(alias_files)
fi
have_models || info "Download a model first: ${C_BOLD}CYWS_MODELS=bf16 ./setup.sh${C_RESET} (or from the web demo, Settings → Model)."
printf '\n   %sNext%s\n' "${C_BOLD}" "${C_RESET}"
((ALIAS_ADDED)) && info "${ALIAS_NAME} works in a new shell (or after ${C_BOLD}source ~/.bashrc${C_RESET}); here, use $(printf '%q' "$HERE/webapp/run.sh")."
printf '   %-52s %s\n' "${run}" "# web demo: https://${ip}:${PORT}/ (and http://)"
printf '   %-52s %s\n' "${run} --daemon" "# the same, in the background"
printf '   %-52s %s\n' "${run} --stop" "# stop it"
here="./"; [[ "$PWD" == "$HERE" ]] || here="$(printf '%q' "$HERE")/"
printf '   %-52s %s\n' "${here}run.sh --mode static --before b.jpg --after a.jpg" "# the CLI, one pair"
printf '   %-52s %s\n' "${here}setup.sh --check" "# load the model on the MLA and run one pair"
printf '   %-52s %s\n' "${run} update" "# the latest version from GitHub"
info "Only one app may use the MLA at a time: stop other demos first."
printf '\n'
