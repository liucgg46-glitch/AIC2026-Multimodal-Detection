from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml


SCRIPT = Path(__file__).parents[1] / "scripts" / "data" / "prepare_ir_yolo.py"
SPEC = importlib.util.spec_from_file_location("prepare_ir_yolo", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_ir(path: Path, value: int = 42, *, channels: int = 3, dtype: np.dtype = np.uint8) -> None:
    shape = (8, 12, channels) if channels > 1 else (8, 12)
    image = np.full(shape, value, dtype=dtype)
    assert cv2.imwrite(str(path), image)


def make_source(
    tmp_path: Path,
    stems: dict[str, str],
    *,
    empty_label: str | None = None,
) -> tuple[Path, Path]:
    ir_dir = tmp_path / "raw" / "train" / "infrared"
    label_dir = tmp_path / "processed" / "train" / "labels_clean"
    ir_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for index, (stem, extension) in enumerate(stems.items()):
        write_ir(ir_dir / f"{stem}{extension}", 20 + index)
        content = "" if stem == empty_label else "0 0.5 0.5 0.1 0.1\n"
        (label_dir / f"{stem}.txt").write_text(content, encoding="utf-8", newline="\n")
    return ir_dir, label_dir


def write_split(path: Path, stems: list[str]) -> Path:
    path.write_text("\n".join(stems) + ("\n" if stems else ""), encoding="utf-8", newline="\n")
    return path


def digest_tree(root: Path) -> list[tuple[str, int, str]]:
    result = []
    for path in sorted((path for path in root.rglob("*") if path.is_file())):
        result.append(
            (
                path.relative_to(root).as_posix(),
                path.stat().st_size,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    return result


def build_small_view(tmp_path: Path, *, mode: str = "copy", output_name: str = "raw3"):
    ir_dir, label_dir = make_source(
        tmp_path,
        {"train_jpg": ".jpg", "train_png": ".png", "val_empty": ".png"},
        empty_label="val_empty",
    )
    train_split = write_split(tmp_path / "train.txt", ["train_jpg", "train_png"])
    val_split = write_split(tmp_path / "val.txt", ["val_empty"])
    output = tmp_path / "ir_trainable" / output_name
    result = MODULE.build_view(
        ir_dir,
        label_dir,
        train_split,
        val_split,
        output,
        link_mode=mode,
        expected_train_count=2,
        expected_val_count=1,
    )
    return ir_dir, label_dir, train_split, val_split, output, result


def test_builds_mixed_extension_raw3_view_with_clean_labels(tmp_path: Path) -> None:
    ir_dir, label_dir, _, _, output, result = build_small_view(tmp_path)

    assert (output / "images" / "train" / "train_jpg.jpg").is_file()
    assert (output / "images" / "train" / "train_png.png").is_file()
    assert (output / "images" / "val" / "val_empty.png").is_file()
    assert (output / "labels" / "val" / "val_empty.txt").read_bytes() == b""
    assert result["validation"]["train_count"] == 2
    assert result["validation"]["val_count"] == 1
    assert result["validation"]["png_count"] == 2
    assert result["validation"]["jpg_count"] == 1

    for source in ir_dir.iterdir():
        subset = "val" if source.stem == "val_empty" else "train"
        assert (output / "images" / subset / source.name).read_bytes() == source.read_bytes()
    for source in label_dir.iterdir():
        subset = "val" if source.stem == "val_empty" else "train"
        assert (output / "labels" / subset / source.name).read_bytes() == source.read_bytes()


def test_data_yaml_has_standard_paths_and_canonical_names(tmp_path: Path) -> None:
    _, _, _, _, output, _ = build_small_view(tmp_path)
    config = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))

    assert Path(config["path"]) == output.resolve()
    assert config["train"] == "images/train"
    assert config["val"] == "images/val"
    assert tuple(config["names"].values()) == MODULE.CLASS_NAMES


def test_split_count_is_strict(tmp_path: Path) -> None:
    ir_dir, label_dir = make_source(tmp_path, {"a": ".png", "b": ".jpg"})
    train = write_split(tmp_path / "train.txt", ["a"])
    val = write_split(tmp_path / "val.txt", ["b"])

    with pytest.raises(MODULE.IRDatasetViewError, match="Split 数量错误"):
        MODULE.validate_sources(ir_dir, label_dir, train, val)


def test_train_val_overlap_is_rejected(tmp_path: Path) -> None:
    ir_dir, label_dir = make_source(tmp_path, {"same": ".png"})
    train = write_split(tmp_path / "train.txt", ["same"])
    val = write_split(tmp_path / "val.txt", ["same"])

    with pytest.raises(MODULE.IRDatasetViewError, match="train/val split 存在重复 stem"):
        MODULE.validate_sources(
            ir_dir, label_dir, train, val, expected_train_count=1, expected_val_count=1
        )


def test_duplicate_stem_inside_split_is_rejected(tmp_path: Path) -> None:
    ir_dir, label_dir = make_source(tmp_path, {"a": ".png", "b": ".jpg"})
    train = write_split(tmp_path / "train.txt", ["a", "a"])
    val = write_split(tmp_path / "val.txt", ["b"])

    with pytest.raises(MODULE.IRDatasetViewError, match="Split 包含重复 stem"):
        MODULE.validate_sources(
            ir_dir, label_dir, train, val, expected_train_count=2, expected_val_count=1
        )


def test_missing_image_is_rejected_before_output(tmp_path: Path) -> None:
    ir_dir, label_dir = make_source(tmp_path, {"train": ".png", "val": ".jpg"})
    (ir_dir / "val.jpg").unlink()
    train = write_split(tmp_path / "train.txt", ["train"])
    val = write_split(tmp_path / "val.txt", ["val"])
    output = tmp_path / "ir_trainable" / "raw3"

    with pytest.raises(MODULE.IRDatasetViewError, match="IR 图像.*缺失"):
        MODULE.build_view(
            ir_dir,
            label_dir,
            train,
            val,
            output,
            expected_train_count=1,
            expected_val_count=1,
        )
    assert not output.exists()


def test_missing_label_is_rejected_before_output(tmp_path: Path) -> None:
    ir_dir, label_dir = make_source(tmp_path, {"train": ".png", "val": ".jpg"})
    (label_dir / "val.txt").unlink()
    train = write_split(tmp_path / "train.txt", ["train"])
    val = write_split(tmp_path / "val.txt", ["val"])

    with pytest.raises(MODULE.IRDatasetViewError, match="labels_clean.*缺失"):
        MODULE.validate_sources(
            ir_dir, label_dir, train, val, expected_train_count=1, expected_val_count=1
        )


def test_duplicate_image_stem_is_rejected(tmp_path: Path) -> None:
    ir_dir, label_dir = make_source(tmp_path, {"train": ".png", "val": ".jpg"})
    write_ir(ir_dir / "train.jpg", 99)
    train = write_split(tmp_path / "train.txt", ["train"])
    val = write_split(tmp_path / "val.txt", ["val"])

    with pytest.raises(MODULE.IRDatasetViewError, match="多个 IR 图像"):
        MODULE.validate_sources(
            ir_dir, label_dir, train, val, expected_train_count=1, expected_val_count=1
        )


@pytest.mark.parametrize("extra_kind", ["image", "label"])
def test_split_external_source_stem_is_rejected(tmp_path: Path, extra_kind: str) -> None:
    ir_dir, label_dir = make_source(tmp_path, {"train": ".png", "val": ".jpg"})
    if extra_kind == "image":
        write_ir(ir_dir / "extra.png", 77)
    else:
        (label_dir / "extra.txt").write_text("", encoding="utf-8")
    train = write_split(tmp_path / "train.txt", ["train"])
    val = write_split(tmp_path / "val.txt", ["val"])

    with pytest.raises(MODULE.IRDatasetViewError, match="split 外额外 stem"):
        MODULE.validate_sources(
            ir_dir, label_dir, train, val, expected_train_count=1, expected_val_count=1
        )


def test_copy_mode_preserves_source_tree(tmp_path: Path) -> None:
    ir_dir, label_dir = make_source(tmp_path, {"train": ".png", "val": ".jpg"})
    source_root = tmp_path / "raw"
    before = digest_tree(source_root)
    train = write_split(tmp_path / "train.txt", ["train"])
    val = write_split(tmp_path / "val.txt", ["val"])
    output = tmp_path / "ir_trainable" / "raw3"

    result = MODULE.build_view(
        ir_dir,
        label_dir,
        train,
        val,
        output,
        link_mode="copy",
        expected_train_count=1,
        expected_val_count=1,
    )

    assert result["manifest"]["link_mode_effective"] == "copy"
    assert digest_tree(source_root) == before


def test_hardlink_mode_creates_hardlinks_when_supported(tmp_path: Path) -> None:
    ir_dir, _, _, _, output, result = build_small_view(tmp_path, mode="hardlink")

    assert result["manifest"]["link_mode_effective"] == "hardlink"
    assert os.path.samefile(ir_dir / "train_jpg.jpg", output / "images" / "train" / "train_jpg.jpg")


def test_auto_falls_back_to_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_link(source: Path, destination: Path) -> None:
        raise OSError("hardlink unavailable")

    monkeypatch.setattr(MODULE.os, "link", fail_link)
    _, _, _, _, _, result = build_small_view(tmp_path, mode="auto")
    assert result["manifest"]["link_mode_effective"] == "copy"


def test_existing_output_is_rejected_without_force(tmp_path: Path) -> None:
    ir_dir, label_dir, train, val, output, _ = build_small_view(tmp_path)
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(MODULE.IRDatasetViewError, match="显式使用 --force"):
        MODULE.build_view(
            ir_dir,
            label_dir,
            train,
            val,
            output,
            expected_train_count=2,
            expected_val_count=1,
        )
    assert marker.read_text(encoding="utf-8") == "keep"


def test_manifest_is_byte_deterministic_and_has_no_staging_path(tmp_path: Path) -> None:
    ir_dir, label_dir = make_source(
        tmp_path, {"train_jpg": ".jpg", "train_png": ".png", "val": ".png"}
    )
    train = write_split(tmp_path / "train.txt", ["train_jpg", "train_png"])
    val = write_split(tmp_path / "val.txt", ["val"])
    manifests = []
    for name in ("first", "second"):
        output = tmp_path / "ir_trainable" / name
        MODULE.build_view(
            ir_dir,
            label_dir,
            train,
            val,
            output,
            link_mode="copy",
            expected_train_count=2,
            expected_val_count=1,
        )
        manifests.append((output / "manifest.json").read_bytes())

    assert manifests[0] == manifests[1]
    assert b"created_at" not in manifests[0]
    assert b".first-" not in manifests[0]
    parsed = json.loads(manifests[0])
    assert parsed["representation"] == "raw3"
    assert parsed["preprocessing"] == "none"
    assert parsed["image_byte_preserving"] is True
    assert list(parsed["source_file_list"]) == sorted(
        parsed["source_file_list"], key=lambda item: item["stem"]
    )
    assert all("\\" not in value for value in (parsed["source_ir_dir"], parsed["label_dir"]))


@pytest.mark.parametrize("representation", ["gray3", "gray1", "normalize3", "clahe3"])
def test_non_raw3_representation_is_explicitly_rejected(tmp_path: Path, representation: str) -> None:
    output = tmp_path / "ir_trainable" / representation
    with pytest.raises(MODULE.IRDatasetViewError, match="仅实现 representation=raw3"):
        MODULE.build_view(
            tmp_path / "unused-ir",
            tmp_path / "unused-labels",
            tmp_path / "unused-train.txt",
            tmp_path / "unused-val.txt",
            output,
            representation=representation,
            expected_train_count=1,
            expected_val_count=1,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("dtype", "channels", "message"),
    [(np.uint16, 3, "uint8"), (np.uint8, 1, "3-channel")],
)
def test_raw3_encoding_contract_is_enforced(
    tmp_path: Path, dtype: np.dtype, channels: int, message: str
) -> None:
    ir_dir, label_dir = make_source(tmp_path, {"train": ".png", "val": ".png"})
    write_ir(ir_dir / "train.png", dtype=dtype, channels=channels)
    train = write_split(tmp_path / "train.txt", ["train"])
    val = write_split(tmp_path / "val.txt", ["val"])

    with pytest.raises(MODULE.IRDatasetViewError, match=message):
        MODULE.validate_sources(
            ir_dir, label_dir, train, val, expected_train_count=1, expected_val_count=1
        )


def test_force_scope_cannot_escape_ir_trainable(tmp_path: Path) -> None:
    allowed = tmp_path / "processed" / "ir_trainable"
    with pytest.raises(MODULE.IRDatasetViewError, match="输出目录必须位于"):
        MODULE.ensure_output_scope(tmp_path / "raw", allowed)


def test_cli_defaults_to_clean_labels_raw3_and_fixed_split(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])
    args = MODULE.parse_args()
    assert args.ir_dir == "data/raw/train/infrared"
    assert args.label_dir == "data/processed/train/labels_clean"
    assert args.train_split == "data/splits/train.txt"
    assert args.val_split == "data/splits/val.txt"
    assert args.output_root == "data/processed/ir_trainable/raw3"
    assert args.representation == "raw3"
    assert args.link_mode == "auto"
