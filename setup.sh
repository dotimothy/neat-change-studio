#!/usr/bin/env bash
# Check the DevKit runtime for the CYWS app: the pyneat venv, numpy, OpenCV with
# SIFT, Pillow, Flask (for the web demo) and the pyneat build. Installs nothing
# unless --install is given (then pip-installs only the missing packages into the venv).
#   ./setup.sh             # check only
#   ./setup.sh --install   # also pip install missing numpy / opencv / pillow / flask
#   ./setup.sh --fetch     # also download the bf16 model from Hugging Face (or --fetch=w8a16, --fetch=all)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
VENV="${PYNEAT_VENV:-$HOME/pyneat}"
INSTALL=0 FETCH=""
for a in "$@"; do
  case "$a" in
    --install) INSTALL=1 ;;
    --fetch) FETCH=bf16 ;;
    --fetch=*) FETCH="${a#--fetch=}" ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

[[ -f "$VENV/bin/activate" ]] || { echo "pyneat environment not found at $VENV (set PYNEAT_VENV)" >&2; exit 1; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"
MISSING_FILE="$(mktemp)"
trap 'rm -f "$MISSING_FILE"' EXIT
export MISSING_FILE
echo "python: $(command -v python3) ($(python3 -c 'import sys; print(sys.version.split()[0])'))"

check() {
  python3 - <<'PY'
import importlib, os, sys
missing = []
for mod, pkg in (("numpy", "numpy"), ("cv2", "opencv-python-headless"), ("PIL", "pillow"), ("flask", "flask")):
    try:
        m = importlib.import_module(mod)
        print(f"{mod}: {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"{mod}: MISSING ({e})")
        missing.append(pkg)
if "opencv-python-headless" not in missing:
    import cv2
    if hasattr(cv2, "SIFT_create"):
        cv2.SIFT_create(100)
        print("cv2.SIFT_create: ok")
    else:
        print("cv2.SIFT_create: missing (registration falls back to ORB)")
try:
    import scipy
    print(f"scipy: {scipy.__version__} (optional)")
except Exception:
    print("scipy: not installed (optional)")
with open(os.environ["MISSING_FILE"], "w") as f:
    f.write(" ".join(missing))
sys.exit(1 if missing else 0)
PY
}

if ! check; then
  missing="$(cat "$MISSING_FILE")"
  if ((INSTALL)) && [[ -n "$missing" ]]; then
    echo "installing: $missing"
    # shellcheck disable=SC2086
    pip install --quiet $missing
    check || { echo "still missing packages after install" >&2; exit 1; }
  else
    echo "missing packages: $missing (re-run with --install to pip install them)" >&2
    exit 1
  fi
fi

python3 - <<'PY' || { echo "pyneat is not importable in $VENV" >&2; exit 1; }
import pyneat
info = pyneat.build_info() if hasattr(pyneat, "build_info") else "(build_info() not available)"
print("pyneat:", info)
PY

if [[ -n "$FETCH" ]]; then
  # shellcheck disable=SC2086
  "$HERE/webapp/run.sh" --models "$HERE/models" --fetch ${FETCH//,/ } || { echo "a model download did not finish; run again to continue it" >&2; exit 1; }
fi
shopt -s nullglob
mpks=("$HERE"/models/*_mpk.tar.gz)
if ((${#mpks[@]})); then for m in "${mpks[@]}"; do echo "model: $m"; done
else echo "model: none in $HERE/models (./setup.sh --fetch downloads bf16 from Hugging Face)"; fi
[[ -d "$HERE/data/testCYWS" ]] && echo "dataset: $HERE/data/testCYWS" || echo "dataset: not deployed (deploy.sh --with-data)"
echo "setup ok"
