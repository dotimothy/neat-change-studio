#!/usr/bin/env bash
# Download a before/after evaluation set from Google Drive, file by file, using a
# manifest you provide. No dataset is bundled with this repo: bring your own pairs.
#
# Manifest: tab-separated "relative_path<TAB>drive_file_id" lines; see
# data/testcyws_manifest.example.tsv. Dataset layout:
#   images/<id>_ref.jpg (before), images/<id>_cap.jpg (after), optional masks/<id>.jpg
# Files are fetched one by one because `gdown --folder` only lists the first 50
# entries of a folder. Re-running skips files that are already present and non-empty.
#
# Usage: data/download_testcyws.sh MANIFEST [DEST]
#        TESTCYWS_MANIFEST=MANIFEST data/download_testcyws.sh [DEST]
# DEST defaults to $CYWS_DATA, or data/testCYWS in the repo.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MANIFEST="${TESTCYWS_MANIFEST:-}"
if [[ -z "$MANIFEST" && $# -gt 0 ]]; then MANIFEST="$1"; shift; fi
if [[ -z "$MANIFEST" || ! -f "$MANIFEST" ]]; then
  {
    [[ -n "$MANIFEST" ]] && echo "manifest not found: $MANIFEST"
    echo "usage: data/download_testcyws.sh MANIFEST [DEST]   (or TESTCYWS_MANIFEST=MANIFEST)"
    echo
    echo "No dataset is bundled with this repo: bring your own before/after pairs."
    echo "Either put them in ${CYWS_DATA:-$HERE/testCYWS} directly"
    echo "(images/<id>_ref.jpg before, images/<id>_cap.jpg after, optional masks/<id>.jpg),"
    echo "or list Google Drive files in a manifest, one per line:"
    echo "  images/<id>_ref.jpg<TAB><drive-file-id>"
    echo "See $HERE/testcyws_manifest.example.tsv."
  } >&2
  exit 1
fi
DEST="${1:-${CYWS_DATA:-$HERE/testCYWS}}"
GDOWN="${GDOWN:-$HERE/../.venv/bin/gdown}"
command -v "$GDOWN" >/dev/null 2>&1 || GDOWN=gdown
command -v "$GDOWN" >/dev/null 2>&1 || { echo "gdown not found (pip install gdown)" >&2; exit 1; }

mkdir -p "$DEST/images" "$DEST/masks"
fail=0 want_img=0 want_mask=0
while IFS=$'\t' read -r rel id; do
  [[ -z "$rel" || "$rel" == \#* ]] && continue
  [[ -n "$id" && "$id" != \<* ]] || { echo "no Drive file ID for $rel" >&2; fail=1; continue; }
  case "$rel" in images/*) want_img=$((want_img + 1)) ;; masks/*) want_mask=$((want_mask + 1)) ;; esac
  out="$DEST/$rel"
  mkdir -p "$(dirname "$out")"
  if [[ -s "$out" ]]; then continue; fi
  for attempt in 1 2 3; do
    if "$GDOWN" --quiet "https://drive.google.com/uc?id=$id" -O "$out.part"; then
      mv "$out.part" "$out"; break
    fi
    echo "retry $attempt: $rel" >&2; sleep $((attempt * 5))
  done
  [[ -s "$out" ]] || { echo "FAILED: $rel" >&2; fail=1; }
done < "$MANIFEST"

n_img=$(find "$DEST/images" -name '*.jpg' | wc -l)
n_mask=$(find "$DEST/masks" -name '*.jpg' | wc -l)
echo "images: $n_img (manifest: $want_img)   masks: $n_mask (manifest: $want_mask)   -> $DEST"
[[ $fail -eq 0 && $n_img -ge $want_img && $n_mask -ge $want_mask ]] || { echo "dataset incomplete" >&2; exit 1; }
