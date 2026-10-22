"""Find before/after(/mask) triplets on disk without decoding anything.

Decoding is left to ingestion.py so that a dataset scan stays cheap and every
problem with the file layout is reported up front, as warnings, before any
image (or the accelerator) is touched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from common import CywsError, PairItem

PathLike = Union[str, Path]

IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png"})
# Filename stem suffix (matched case-insensitively) -> role.
_ROLE_SUFFIXES = (("_ref", "before"), ("_cap", "after"), ("_mask", "mask"))


def _image_files(directory: Path) -> list[Path]:
    """Image files directly in `directory`, sorted by name, hidden files skipped."""
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTS
    )


def _split_role(path: Path) -> tuple[str, str]:
    """Return (pair_id, role) where role is before / after / mask / plain."""
    stem = path.stem
    lower = stem.lower()
    for suffix, role in _ROLE_SUFFIXES:
        if lower.endswith(suffix) and len(stem) > len(suffix):
            return stem[: -len(suffix)], role
    return stem, "plain"


def _put(table: dict[str, Path], pair_id: str, path: Path, what: str, warnings: list[str]) -> None:
    """Insert keeping the first file seen; later duplicates become warnings."""
    if pair_id in table:
        warnings.append(
            f"duplicate {what} for '{pair_id}': using {table[pair_id]}, ignoring {path}"
        )
    else:
        table[pair_id] = path


def discover_pairs(root: PathLike) -> tuple[list[PairItem], list[str]]:
    """Scan a dataset directory for <id>_ref / <id>_cap pairs and optional masks.

    Layout: root/images/<id>_ref.<ext>, root/images/<id>_cap.<ext>,
    root/masks/<id>.<ext>; the same files may also sit directly in root (flat
    layout, where a mask is <id>.<ext> or <id>_mask.<ext>). Extensions are
    jpg/jpeg/png in any case. Files in images/ win over duplicates in root and
    masks/ wins over flat masks.

    Returns the complete pairs sorted by id, and human-readable warnings for
    everything that was skipped (orphans, duplicates, unrecognised names).
    """
    root = Path(root).expanduser()
    if not root.exists():
        raise CywsError(f"dataset directory does not exist: {root}")
    if not root.is_dir():
        raise CywsError(f"dataset path is not a directory: {root}")

    warnings: list[str] = []
    befores: dict[str, Path] = {}
    afters: dict[str, Path] = {}
    masks: dict[str, Path] = {}
    plain: list[Path] = []

    masks_dir = root / "masks"
    if masks_dir.is_dir():
        for path in _image_files(masks_dir):
            pair_id, role = _split_role(path)
            if role in ("before", "after"):
                warnings.append(f"ignoring {role} image inside masks/: {path}")
                continue
            _put(masks, pair_id, path, "mask", warnings)

    image_dirs = [d for d in (root / "images", root) if d.is_dir()]
    for directory in image_dirs:
        for path in _image_files(directory):
            pair_id, role = _split_role(path)
            if role == "before":
                _put(befores, pair_id, path, "before (_ref) image", warnings)
            elif role == "after":
                _put(afters, pair_id, path, "after (_cap) image", warnings)
            elif role == "mask":
                if pair_id not in masks:
                    masks[pair_id] = path
                elif masks[pair_id] != path:
                    warnings.append(
                        f"duplicate mask for '{pair_id}': using {masks[pair_id]}, ignoring {path}"
                    )
            else:
                plain.append(path)

    complete = sorted(set(befores) & set(afters))
    complete_set = set(complete)

    # A suffix-less image next to the pairs is a flat-layout mask only when its
    # stem names a complete pair; anything else is unrelated and reported.
    for path in plain:
        if path.stem in complete_set and path.stem not in masks:
            masks[path.stem] = path
        elif path.stem in complete_set:
            warnings.append(
                f"duplicate mask for '{path.stem}': using {masks[path.stem]}, ignoring {path}"
            )
        else:
            warnings.append(f"ignoring image without a _ref/_cap/_mask suffix: {path}")

    for pair_id in sorted(set(befores) - complete_set):
        warnings.append(f"'{pair_id}': before image {befores[pair_id]} has no _cap (after) image; skipped")
    for pair_id in sorted(set(afters) - complete_set):
        warnings.append(f"'{pair_id}': after image {afters[pair_id]} has no _ref (before) image; skipped")
    for pair_id in sorted(set(masks) - complete_set):
        warnings.append(f"'{pair_id}': mask {masks[pair_id]} has no complete before/after pair; ignored")

    if not complete:
        looked = ", ".join(str(d) for d in image_dirs)
        raise CywsError(
            f"no complete <id>_ref / <id>_cap image pair found (looked in {looked})"
        )

    items = [
        PairItem(pair_id=i, before_path=befores[i], after_path=afters[i], mask_path=masks.get(i))
        for i in complete
    ]
    return items, warnings


def _existing_file(path: PathLike, what: str) -> Path:
    p = Path(path).expanduser()
    if not p.exists():
        raise CywsError(f"{what} image does not exist: {p}")
    if not p.is_file():
        raise CywsError(f"{what} path is not a file: {p}")
    return p


def static_pair(
    before: PathLike,
    after: PathLike,
    mask: Optional[PathLike] = None,
    pair_id: Optional[str] = None,
) -> PairItem:
    """Build a PairItem from explicit paths, failing early on missing files.

    pair_id defaults to the after file's stem, since the after image defines the
    output frame and naming.
    """
    before_p = _existing_file(before, "before")
    after_p = _existing_file(after, "after")
    mask_p = _existing_file(mask, "mask") if mask is not None else None
    pid = after_p.stem if pair_id is None else str(pair_id).strip()
    if not pid:
        raise CywsError("pair_id must not be empty")
    if "/" in pid or "\\" in pid:
        raise CywsError(f"pair_id must not contain path separators: {pid!r}")
    return PairItem(pair_id=pid, before_path=before_p, after_path=after_p, mask_path=mask_p)
