from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import yaml


SCRIPT = Path(__file__).parents[1] / "scripts" / "train" / "prepare_rgb_yolo.py"
SPEC = importlib.util.spec_from_file_location("prepare_rgb_yolo", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_source(tmp_path: Path, stems: dict[str, str]) -> Path:
    root = tmp_path / "raw" / "train"
    (root / "visible").mkdir(parents=True)
    (root / "labels").mkdir()
    for stem, extension in stems.items():
        (root / "visible" / f"{stem}{extension}").write_bytes(stem.encode())
        (root / "labels" / f"{stem}.txt").write_text(
            "0 0.5 0.5 0.1 0.1\n", encoding="utf-8"
        )
    return root


def write_split(path: Path, stems: list[str]) -> Path:
    path.write_text("\n".join(stems) + ("\n" if stems else ""), encoding="utf-8")
    return path


def test_builds_standard_view_for_mixed_extensions(tmp_path: Path) -> None:
    data_root = make_source(tmp_path, {"sample_a": ".jpg", "sample_b": ".png"})
    train_split = write_split(tmp_path / "train.txt", ["sample_a"])
    val_split = write_split(tmp_path / "val.txt", ["sample_b"])
    output = tmp_path / "processed" / "rgb_yolo"

    methods = MODULE.build_view(data_root, train_split, val_split, output)

    assert (output / "images" / "train" / "sample_a.jpg").is_file()
    assert (output / "images" / "val" / "sample_b.png").is_file()
    assert (output / "labels" / "train" / "sample_a.txt").is_file()
    assert (output / "labels" / "val" / "sample_b.txt").is_file()
    assert sum(methods.values()) == 4
    if methods["hardlink"]:
        assert os.path.samefile(
            data_root / "visible" / "sample_a.jpg",
            output / "images" / "train" / "sample_a.jpg",
        )

    config = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert "path" not in config
    assert config["train"] == "images/train"
    assert config["val"] == "images/val"
    assert len(config["names"]) == 12


def test_builds_view_from_raw_visible_and_independent_clean_labels(
    tmp_path: Path,
) -> None:
    data_root = make_source(tmp_path, {"train_sample": ".jpg", "val_sample": ".png"})
    label_dir = tmp_path / "processed" / "train" / "labels_clean"
    label_dir.mkdir(parents=True)
    clean_labels = {
        "train_sample": "6 0.4 0.4 0.2 0.2\n",
        "val_sample": "8 0.6 0.6 0.1 0.1\n",
    }
    for stem, content in clean_labels.items():
        (label_dir / f"{stem}.txt").write_text(content, encoding="utf-8")

    train_split = write_split(tmp_path / "train.txt", ["train_sample"])
    val_split = write_split(tmp_path / "val.txt", ["val_sample"])
    output = tmp_path / "processed" / "rgb_yolo"

    MODULE.build_view(
        data_root,
        train_split,
        val_split,
        output,
        label_dir=label_dir,
    )

    assert (output / "images" / "train" / "train_sample.jpg").read_bytes() == b"train_sample"
    assert (output / "images" / "val" / "val_sample.png").read_bytes() == b"val_sample"
    assert (output / "labels" / "train" / "train_sample.txt").read_text(
        encoding="utf-8"
    ) == clean_labels["train_sample"]
    assert (output / "labels" / "val" / "val_sample.txt").read_text(
        encoding="utf-8"
    ) == clean_labels["val_sample"]


def test_empty_split_stops_without_creating_output(tmp_path: Path) -> None:
    data_root = make_source(tmp_path, {"sample": ".jpg"})
    train_split = write_split(tmp_path / "train.txt", [])
    val_split = write_split(tmp_path / "val.txt", ["sample"])
    output = tmp_path / "processed" / "rgb_yolo"

    with pytest.raises(MODULE.DatasetViewError, match="拒绝回退到全部训练数据"):
        MODULE.build_view(data_root, train_split, val_split, output)

    assert not output.exists()


def test_duplicate_image_stem_is_rejected(tmp_path: Path) -> None:
    data_root = make_source(tmp_path, {"sample": ".jpg"})
    (data_root / "visible" / "sample.png").write_bytes(b"duplicate")
    train_split = write_split(tmp_path / "train.txt", ["sample"])
    val_split = write_split(tmp_path / "val.txt", ["other"])

    with pytest.raises(MODULE.DatasetViewError, match="匹配到多个 RGB 图像"):
        MODULE.validate_sources(data_root, train_split, val_split)


def test_train_val_overlap_is_rejected(tmp_path: Path) -> None:
    data_root = make_source(tmp_path, {"sample": ".png"})
    train_split = write_split(tmp_path / "train.txt", ["sample"])
    val_split = write_split(tmp_path / "val.txt", ["sample"])

    with pytest.raises(MODULE.DatasetViewError, match="train/val split 存在重复 stem"):
        MODULE.validate_sources(data_root, train_split, val_split)
