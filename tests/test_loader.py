"""dataset_loader: layout discovery, orphans, sorting, static pairs."""

from pathlib import Path

import pytest

from common import CywsError
from dataset_loader import discover_pairs, static_pair


def touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")  # discovery never decodes
    return path


def test_images_masks_layout_sorted(tmp_path):
    for pid in ("b", "c", "a"):
        touch(tmp_path / "images" / f"{pid}_ref.jpg")
        touch(tmp_path / "images" / f"{pid}_cap.jpg")
    touch(tmp_path / "masks" / "a.jpg")
    touch(tmp_path / "masks" / "c.png")

    items, warnings = discover_pairs(tmp_path)

    assert [i.pair_id for i in items] == ["a", "b", "c"]
    assert warnings == []
    a, b, c = items
    assert a.before_path == tmp_path / "images" / "a_ref.jpg"
    assert a.after_path == tmp_path / "images" / "a_cap.jpg"
    assert a.mask_path == tmp_path / "masks" / "a.jpg"
    assert b.mask_path is None
    assert c.mask_path == tmp_path / "masks" / "c.png"


def test_orphans_are_reported(tmp_path):
    touch(tmp_path / "images" / "ok_ref.jpg")
    touch(tmp_path / "images" / "ok_cap.jpg")
    touch(tmp_path / "images" / "lonely_ref.jpg")
    touch(tmp_path / "images" / "solo_cap.png")
    touch(tmp_path / "masks" / "ghost.jpg")
    touch(tmp_path / "masks" / "solo.jpg")  # mask whose pair is incomplete

    items, warnings = discover_pairs(tmp_path)

    assert [i.pair_id for i in items] == ["ok"]
    text = "\n".join(warnings)
    assert "lonely" in text and "_cap" in text
    assert "solo_cap.png" in text and "_ref" in text
    assert "ghost.jpg" in text
    assert "solo.jpg" in text
    assert len(warnings) == 4


def test_mixed_extensions_and_case(tmp_path):
    touch(tmp_path / "images" / "x1_REF.JPG")
    touch(tmp_path / "images" / "x1_Cap.png")
    touch(tmp_path / "images" / "x2_ref.jpeg")
    touch(tmp_path / "images" / "x2_cap.JPEG")
    touch(tmp_path / "images" / "notes.txt")       # ignored silently
    touch(tmp_path / "images" / ".hidden_ref.jpg")  # ignored silently
    touch(tmp_path / "masks" / "x1.PNG")

    items, warnings = discover_pairs(tmp_path)

    assert [i.pair_id for i in items] == ["x1", "x2"]
    assert items[0].before_path.name == "x1_REF.JPG"
    assert items[0].after_path.name == "x1_Cap.png"
    assert items[0].mask_path.name == "x1.PNG"
    assert items[1].mask_path is None
    assert warnings == []


def test_flat_layout(tmp_path):
    touch(tmp_path / "p_ref.jpg")
    touch(tmp_path / "p_cap.jpg")
    touch(tmp_path / "p.png")          # flat mask, plain stem
    touch(tmp_path / "q_ref.jpg")
    touch(tmp_path / "q_cap.jpg")
    touch(tmp_path / "q_mask.jpg")     # flat mask, explicit suffix
    touch(tmp_path / "holiday.jpg")    # unrelated image

    items, warnings = discover_pairs(tmp_path)

    assert [i.pair_id for i in items] == ["p", "q"]
    assert items[0].mask_path == tmp_path / "p.png"
    assert items[1].mask_path == tmp_path / "q_mask.jpg"
    assert len(warnings) == 1 and "holiday.jpg" in warnings[0]


def test_duplicate_extension_keeps_first_and_warns(tmp_path):
    touch(tmp_path / "images" / "d_ref.jpg")
    touch(tmp_path / "images" / "d_ref.png")
    touch(tmp_path / "images" / "d_cap.jpg")

    items, warnings = discover_pairs(tmp_path)

    assert items[0].before_path.name == "d_ref.jpg"
    assert len(warnings) == 1 and "duplicate" in warnings[0] and "d_ref.png" in warnings[0]


def test_masks_dir_wins_over_flat_mask(tmp_path):
    touch(tmp_path / "m_ref.jpg")
    touch(tmp_path / "m_cap.jpg")
    touch(tmp_path / "m.jpg")
    touch(tmp_path / "masks" / "m.png")

    items, warnings = discover_pairs(tmp_path)

    assert items[0].mask_path == tmp_path / "masks" / "m.png"
    assert len(warnings) == 1 and "duplicate mask" in warnings[0]


def test_missing_root_raises(tmp_path):
    with pytest.raises(CywsError, match="does not exist"):
        discover_pairs(tmp_path / "nope")


def test_root_is_file_raises(tmp_path):
    f = touch(tmp_path / "file.jpg")
    with pytest.raises(CywsError, match="not a directory"):
        discover_pairs(f)


def test_no_complete_pair_raises(tmp_path):
    touch(tmp_path / "images" / "a_ref.jpg")
    touch(tmp_path / "images" / "b_cap.jpg")
    with pytest.raises(CywsError, match="no complete"):
        discover_pairs(tmp_path)


def test_static_pair_defaults_and_validation(tmp_path):
    before = touch(tmp_path / "before.jpg")
    after = touch(tmp_path / "scene_after.png")
    mask = touch(tmp_path / "gt.png")

    item = static_pair(before, after)
    assert item.pair_id == "scene_after"
    assert item.before_path == before and item.after_path == after and item.mask_path is None

    item = static_pair(str(before), str(after), mask=str(mask), pair_id="custom")
    assert item.pair_id == "custom" and item.mask_path == mask

    with pytest.raises(CywsError, match="does not exist"):
        static_pair(tmp_path / "missing.jpg", after)
    with pytest.raises(CywsError, match="not a file"):
        static_pair(before, tmp_path)
    with pytest.raises(CywsError, match="mask"):
        static_pair(before, after, mask=tmp_path / "missing_mask.png")
    with pytest.raises(CywsError, match="empty"):
        static_pair(before, after, pair_id="  ")
