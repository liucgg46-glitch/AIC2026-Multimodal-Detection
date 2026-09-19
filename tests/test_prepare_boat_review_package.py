"""Portable boat review package preserves source bytes and exact label row indices."""

import csv
import json

import scripts.analysis.prepare_boat_review_package as REVIEW


def test_build_package_is_read_only_and_keeps_physical_row_indices(tmp_path, monkeypatch):
    monkeypatch.setattr(REVIEW, "TRAIN_COUNT", 1)
    monkeypatch.setattr(REVIEW, "VAL_COUNT", 1)
    monkeypatch.setattr(REVIEW, "BOAT_IMAGE_COUNT", 2)
    monkeypatch.setattr(REVIEW, "BOAT_BOX_COUNT", 2)
    labels = tmp_path / "labels_clean"
    visible = tmp_path / "visible"
    r8 = tmp_path / "r8"
    for folder in (labels, visible, r8):
        folder.mkdir()
    (labels / "train.txt").write_text("0 0.2 0.2 0.1 0.1\n1 0.5 0.5 0.2 0.2\n")
    (labels / "val.txt").write_text("1 0.4 0.4 0.1 0.1\n")
    (visible / "train.png").write_bytes(b"train image bytes")
    (visible / "val.png").write_bytes(b"val image bytes")
    train_split = tmp_path / "train_split.txt"
    val_split = tmp_path / "val_split.txt"
    train_split.write_text("train\n")
    val_split.write_text("val\n")
    (r8 / "summary.json").write_text('{"images":1}')
    with (r8 / "gt_misses_at_conf_0p001.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "stem", "class_id", "x", "y", "w", "h", "reason",
            "other_class_id", "other_class_iou",
        ])
        writer.writeheader()
        writer.writerow({"stem": "val", "class_id": 1, "x": 0.4, "y": 0.4,
                         "w": 0.1, "h": 0.1, "reason": "other_class_candidate",
                         "other_class_id": 0, "other_class_iou": 0.8})
    template = tmp_path / "template.html"
    template.write_text("<script>__REVIEW_DATA_JSON__</script>")
    output = tmp_path / "package"
    original_label = (labels / "train.txt").read_bytes()
    manifest = REVIEW.build_package(output, labels, visible, train_split, val_split, r8, template)
    assert manifest["image_count"] == 2 and manifest["boat_box_count"] == 2
    assert (labels / "train.txt").read_bytes() == original_label
    assert (output / "labels_clean_snapshot/train.txt").read_bytes() == original_label
    assert (output / "images/train.png").read_bytes() == b"train image bytes"
    with (output / "review.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert rows[0]["stem"] == "train" and rows[0]["row_index"] == "2"
    assert rows[0]["original_bbox"] == "0.5 0.5 0.2 0.2"
    assert rows[0]["decision"] == "unreviewed" and rows[0]["proposed_class"] == ""
    assert rows[1]["r8_reason"] == "other_class_candidate"
    assert '"row_index": 2' in (output / "index.html").read_text()
    assert json.loads((output / "manifest.json").read_text())["val_images"] == 1
