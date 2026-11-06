#!/usr/bin/env python3
"""Download Change Studio's test pairs: before/after photos from PASLCD (MIT License).

PASLCD, the Pose-Agnostic Scene-Level Change Detection dataset (Galappaththige et al.,
CVPR 2025), has ten indoor and outdoor scenes photographed before and after objects were
moved, added or removed, with hand-drawn change masks. data/paslcd_pairs.tsv says which
photos make up each pair; this fetches only those (about 60 photos, not the 31 GB set),
scales them to the masks' 1600-pixel width, and writes the layout Change Studio reads:

    DEST/images/<id>_ref.jpg   before
    DEST/images/<id>_cap.jpg   after
    DEST/masks/<id>.png        the change (all black for a pair where nothing changed)

Usage: data/fetch_paslcd.py [DEST]     (default: data/paslcd in the repo)
Files already there are kept, so a second run only fills in what is missing.
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path

from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
PAIRS = HERE / "paslcd_pairs.tsv"
REPO = "ChamudithaJay/PASLCD"
REVISION = "aef858feb483dd81104fbb5232a8c98fbf4d3fcc"     # pinned, so the pairs never shift
WIDTH = 1600                                              # the width of PASLCD's masks

NOTICE = """Test pairs for Neat Change Studio, from PASLCD
=============================================

Pose-Agnostic Scene-Level Change Detection dataset (PASLCD), by Chamuditha Jayanga
Galappaththige, Jason Lai, Lloyd Windrim, Donald Dansereau, Niko Suenderhauf and Dimity
Miller: https://huggingface.co/datasets/ChamudithaJay/PASLCD (revision {rev}),
released under the MIT License.

  Galappaththige et al., "Multi-View Pose-Agnostic Change Localization with Zero Labels",
  CVPR 2025, pp. 11600-11610.

Change Studio uses {n} pairs: photos scaled to {w} pixels wide, an "after" photo with the
"before" photo from the nearest viewpoint (or two "before" photos for a pair where nothing
changed), and the dataset's change masks. data/paslcd_pairs.tsv lists them.
"""


def pairs() -> list[dict]:
    rows = []
    for line in PAIRS.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            pid, kind, scene, inst, before, after = line.split("\t")
            rows.append({"id": pid, "kind": kind, "scene": scene, "instance": inst, "before": before, "after": after})
    return rows


def get(path: str, timeout: float = 120) -> bytes:
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "neat-change-studio"})
    last = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except OSError as e:
            last = e
    raise OSError(f"could not download {url}: {last}")


def scaled(data: bytes, size: tuple[int, int] | None = None, mode: str = "RGB") -> Image.Image:
    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im).convert(mode)
    if size is None:
        size = (WIDTH, round(im.height * WIDTH / im.width))
    return im.resize(size, Image.NEAREST if mode == "L" else Image.LANCZOS)


def save(im: Image.Image, dest: Path) -> None:
    part = dest.with_name(dest.name + ".part")
    if dest.suffix == ".png":
        im.save(part, "PNG")
    else:
        im.save(part, "JPEG", quality=92)
    part.replace(dest)


def fetch(dest: Path, log=print) -> int:
    (dest / "images").mkdir(parents=True, exist_ok=True)
    (dest / "masks").mkdir(parents=True, exist_ok=True)
    rows = pairs()
    done = 0
    for p in rows:
        base = f"{p['scene']}/Instance_{p['instance']}"
        ref, cap, mask = (dest / "images" / f"{p['id']}_ref.jpg", dest / "images" / f"{p['id']}_cap.jpg",
                          dest / "masks" / f"{p['id']}.png")
        if ref.is_file() and cap.is_file() and mask.is_file():
            done += 1
            continue
        after = scaled(get(f"{base}/images/{p['after']}"))
        before = scaled(get(f"{base}/images/{p['before']}"), after.size)
        if p["kind"] == "change":
            m = scaled(get(f"{base}/gt_mask/{Path(p['after']).stem}.png"), after.size, "L")
            m = m.point(lambda v: 255 if v > 127 else 0)
        else:
            m = Image.new("L", after.size, 0)
        save(before, ref)
        save(after, cap)
        save(m, mask)
        done += 1
        log(f"  {p['id']}")
    (dest / "NOTICE.txt").write_text(NOTICE.format(rev=REVISION, n=len(rows), w=WIDTH))
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dest", nargs="?", type=Path, default=HERE / "paslcd")
    args = ap.parse_args()
    try:
        n = fetch(args.dest)
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"{n} test pairs in {args.dest} (PASLCD, MIT License)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
