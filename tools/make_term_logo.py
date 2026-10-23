#!/usr/bin/env python3
"""Draw the app logo for the terminal: truecolor half blocks (▀), two pixels per character.

    python3 tools/make_term_logo.py                 # webapp/term_logo.txt, 18 columns
    python3 tools/make_term_logo.py --cols 22 --preview /tmp/logo.png

webapp/run.sh prints the file next to the name when the terminal shows colour (term.sh).
Each line is the escape codes for one row of characters, ending in a colour reset; fully
transparent pixels stay the terminal's own background. --preview draws the result as an image.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def pixels(src: Path, cols: int) -> np.ndarray:
    img = Image.open(src).convert("RGBA")
    a = np.asarray(img, np.float32) / 255
    a[..., :3] *= a[..., 3:4]                       # premultiply, so edges don't go dark
    small = Image.fromarray((a * 255).astype(np.uint8), "RGBA").resize((cols, cols), Image.LANCZOS)
    p = np.asarray(small, np.float32) / 255
    rgb = np.where(p[..., 3:4] > 0.02, p[..., :3] / np.maximum(p[..., 3:4], 1e-3), 0)
    return np.dstack([np.clip(rgb, 0, 1), p[..., 3]])


def lines(p: np.ndarray, alpha_cut: float = 0.5) -> list[str]:
    out = []
    for y in range(0, p.shape[0], 2):
        row = ""
        for x in range(p.shape[1]):
            top, bot = p[y, x], p[y + 1, x] if y + 1 < p.shape[0] else np.zeros(4)
            t = tuple(int(v * 255) for v in top[:3]); b = tuple(int(v * 255) for v in bot[:3])
            ton, bon = top[3] >= alpha_cut, bot[3] >= alpha_cut
            if ton and bon:
                row += f"\033[38;2;{t[0]};{t[1]};{t[2]}m\033[48;2;{b[0]};{b[1]};{b[2]}m▀"
            elif ton:
                row += f"\033[49m\033[38;2;{t[0]};{t[1]};{t[2]}m▀"
            elif bon:
                row += f"\033[49m\033[38;2;{b[0]};{b[1]};{b[2]}m▄"
            else:
                row += "\033[0m "
        out.append(row + "\033[0m")
    return out


def preview(p: np.ndarray, dst: Path, cell=(8, 16), bg=(18, 20, 24)) -> None:
    h, w = p.shape[:2]
    img = np.zeros((h // 2 * cell[1], w * cell[0], 3), np.uint8) + np.array(bg, np.uint8)
    for y in range(0, h - 1, 2):
        for x in range(w):
            for k, half in enumerate((p[y, x], p[y + 1, x])):
                if half[3] >= 0.5:
                    img[y // 2 * cell[1] + k * cell[1] // 2: y // 2 * cell[1] + (k + 1) * cell[1] // 2,
                        x * cell[0]:(x + 1) * cell[0]] = (half[:3] * 255).astype(np.uint8)
    Image.fromarray(img).save(dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=ROOT / "webapp/web/assets/neat-change-512.png")
    ap.add_argument("--cols", type=int, default=18)
    ap.add_argument("--out", type=Path, default=ROOT / "webapp/term_logo.txt")
    ap.add_argument("--preview", type=Path)
    a = ap.parse_args()
    p = pixels(a.src, a.cols)
    if a.preview:
        preview(p, a.preview)
    else:
        a.out.write_text("\n".join(lines(p)) + "\n")
        print(f"wrote {a.out} ({a.cols} columns x {len(lines(p))} rows)")
