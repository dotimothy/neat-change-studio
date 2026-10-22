#!/usr/bin/env bash
# Fetch upstream CYWS sources (for the numeric reference check) and the
# pretrained checkpoint. Everything lands in third_party/ and build/, both
# gitignored and never redistributed by this repository.
# CYWS is by Ragav Sachdeva and Andrew Zisserman (The Change You Want to See,
# WACV 2023, https://github.com/ragavsachdeva/The-Change-You-Want-to-See). The
# authors have confirmed to the maintainer (2026) that CYWS, its code and
# pretrained weights, is released under the Apache License 2.0. This supersedes
# the CC BY-NC default mentioned in upstream issue #11 (2024). The upstream
# repository does not include a LICENSE file yet.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="$ROOT/third_party"; CK="$ROOT/build/ckpt"
mkdir -p "$TP" "$CK"

CYWS_REPO=https://github.com/ragavsachdeva/The-Change-You-Want-to-See.git
SMP_REPO=https://github.com/ragavsachdeva/segmentation_models.pytorch.git
SMP_SHA=0092ee4d6f851d89a4a401bb2dfa6187660b8dd3
CKPT_URL=https://thor.robots.ox.ac.uk/~vgg/data/cyws/pretrained-resnet50-3x-coam-scSE-affine.ckpt.gz
SUMS_URL=https://thor.robots.ox.ac.uk/~vgg/data/cyws/SHA512SUMS
CKPT=pretrained-resnet50-3x-coam-scSE-affine.ckpt

clone_at() {  # url dir [sha]
  local url=$1 dir=$2 sha=${3:-}
  [[ -d "$dir/.git" ]] || git clone --quiet "$url" "$dir"
  if [[ -n "$sha" ]]; then git -C "$dir" checkout --quiet "$sha"; fi
  echo "$(basename "$dir") @ $(git -C "$dir" rev-parse HEAD)"
}
clone_at "$CYWS_REPO" "$TP/cyws"
clone_at "$SMP_REPO" "$TP/smp" "$SMP_SHA"

if [[ ! -s "$CK/$CKPT" ]]; then
  curl -fL --retry 3 -o "$CK/$CKPT.gz" "$CKPT_URL"
  curl -fsSL -o "$CK/SHA512SUMS" "$SUMS_URL"
  ( cd "$CK" && grep " $CKPT.gz\$\| \*$CKPT.gz\$" SHA512SUMS | sha512sum -c - )
  gzip -d "$CK/$CKPT.gz"
fi
sha256sum "$CK/$CKPT" | tee "$CK/$CKPT.sha256"
