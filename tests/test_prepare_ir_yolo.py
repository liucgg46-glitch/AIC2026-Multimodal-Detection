from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest
import yaml


SCRIPT = Path(__file__).parents[1] / "scripts" / "data" / "prepare_ir_yolo.py"
REPO_ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_ir_yolo", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def write_ir(
    path: Path,
    value: int = 42,
    channels: int = 3,
    dtype: np.dtype = np.uint8,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (8, 12, channels) if channels > 1 else (8, 12)
    image = np.full(shape, value, dtype=dtype)
    assert cv2.imwrite(str(path), image)


def digest_tree(root: Path) -> List[Tuple[str, int, str]]:
    result = []
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        result.append(
            (
                path.relative_to(root).as_posix(),
                path.stat().st_size,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    return result


def configure_contract_hashes(monkeypatch: pytest.MonkeyPatch, paths: Dict[str, Path]) -> None:
    monkeypatch.setattr(
        MODULE,
        "CANONICAL_TRAIN_SPLIT_SHA256",
        MODULE.sha256_file(paths["train_split"]),
    )
    monkeypatch.setattr(
        MODULE,
        "CANONICAL_VAL_SPLIT_SHA256",
        MODULE.sha256_file(paths["val_split"]),
    )
    labels = list(MODULE.index_labels(paths["label_dir"]).values())
    identity, _ = MODULE._aggregate_paths(labels)
    monkeypatch.setattr(
        MODULE,
        "CANONICAL_LABELS_CLEAN_SHA256",
        identity["aggregate_sha256"],
    )


@pytest.fixture
def canonical_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Dict[str, Path]:
    root = tmp_path / "repo"
    ir_dir = root / MODULE.CANONICAL_IR_RELATIVE
    label_dir = root / MODULE.CANONICAL_LABEL_RELATIVE
    raw_labels = root / "data/raw/train/labels"
    deprecated_labels = raw_labels / "labels"
    train_split = root / MODULE.CANONICAL_TRAIN_SPLIT_RELATIVE
    val_split = root / MODULE.CANONICAL_VAL_SPLIT_RELATIVE
    output = root / MODULE.CANONICAL_OUTPUT_RELATIVE

    for directory in (
        ir_dir,
        label_dir,
        raw_labels,
        deprecated_labels,
        output.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    extensions = {"train_jpg": ".jpg", "train_png": ".png", "val_empty": ".png"}
    for index, (stem, extension) in enumerate(extensions.items()):
        write_ir(ir_dir / f"{stem}{extension}", 20 + index)
        label = "" if stem == "val_empty" else "0 0.5 0.5 0.1 0.1\n"
        write_text_lf(label_dir / f"{stem}.txt", label)
    write_text_lf(train_split, "train_jpg\ntrain_png\n")
    write_text_lf(val_split, "val_empty\n")

    monkeypatch.setattr(MODULE, "PROJECT_ROOT", root)
    paths = {
        "root": root,
        "ir_dir": ir_dir,
        "label_dir": label_dir,
        "raw_labels": raw_labels,
        "deprecated_labels": deprecated_labels,
        "train_split": train_split,
        "val_split": val_split,
        "output": output,
    }
    configure_contract_hashes(monkeypatch, paths)
    return paths


def build_canonical(
    paths: Dict[str, Path],
    link_mode: str = "copy",
    force: bool = False,
) -> Dict[str, object]:
    return MODULE.build_view(
        paths["ir_dir"],
        paths["label_dir"],
        paths["train_split"],
        paths["val_split"],
        paths["output"],
        link_mode=link_mode,
        force=force,
        expected_train_count=2,
        expected_val_count=1,
    )


def test_official_fixed_split_sha_constants_match_repository() -> None:
    assert MODULE.CANONICAL_TRAIN_SPLIT_SHA256 == (
        "f34d3edae8ebd182a258fc7a03d200e313b2635c18831360ae72bd9cb29cb64e"
    )
    assert MODULE.CANONICAL_VAL_SPLIT_SHA256 == (
        "e6d7222de1bd8eb1c1f63346d32fdbeccb0121b21ffcd0777e55961b3d2156d9"
    )
    assert MODULE.sha256_file(REPO_ROOT / "data/splits/train.txt") == MODULE.CANONICAL_TRAIN_SPLIT_SHA256
    assert MODULE.sha256_file(REPO_ROOT / "data/splits/val.txt") == MODULE.CANONICAL_VAL_SPLIT_SHA256


def test_canonical_labels_and_splits_build_successfully(
    canonical_project: Dict[str, Path],
) -> None:
    result = build_canonical(canonical_project)
    assert result["validation"]["train_count"] == 2
    assert result["validation"]["val_count"] == 1
    assert result["validation"]["total_count"] == 3


@pytest.mark.parametrize("label_kind", ["raw_labels", "deprecated_labels", "custom"])
def test_noncanonical_label_directories_are_rejected_clearly(
    canonical_project: Dict[str, Path],
    label_kind: str,
) -> None:
    if label_kind == "custom":
        label_dir = canonical_project["root"] / "custom-labels"
        label_dir.mkdir()
    else:
        label_dir = canonical_project[label_kind]
    with pytest.raises(MODULE.IRDatasetViewError, match="label-dir 必须使用项目 canonical 路径"):
        MODULE.build_view(
            canonical_project["ir_dir"],
            label_dir,
            canonical_project["train_split"],
            canonical_project["val_split"],
            canonical_project["output"],
            expected_train_count=2,
            expected_val_count=1,
        )


@pytest.mark.parametrize("split_name", ["train_split", "val_split"])
def test_alternative_split_path_is_rejected_even_with_same_bytes(
    canonical_project: Dict[str, Path],
    split_name: str,
) -> None:
    replacement = canonical_project["root"] / f"alternative-{split_name}.txt"
    replacement.write_bytes(canonical_project[split_name].read_bytes())
    train = replacement if split_name == "train_split" else canonical_project["train_split"]
    val = replacement if split_name == "val_split" else canonical_project["val_split"]
    with pytest.raises(MODULE.IRDatasetViewError, match="必须使用项目 canonical 路径"):
        MODULE.build_view(
            canonical_project["ir_dir"],
            canonical_project["label_dir"],
            train,
            val,
            canonical_project["output"],
            expected_train_count=2,
            expected_val_count=1,
        )


@pytest.mark.parametrize("split_name", ["train_split", "val_split"])
def test_canonical_split_content_tampering_is_rejected_by_sha(
    canonical_project: Dict[str, Path],
    split_name: str,
) -> None:
    with canonical_project[split_name].open("a", encoding="utf-8", newline="\n") as stream:
        stream.write("tampered\n")
    expected = "train split SHA-256" if split_name == "train_split" else "val split SHA-256"
    with pytest.raises(MODULE.IRDatasetViewError, match=expected):
        MODULE.validate_canonical_inputs(
            canonical_project["label_dir"],
            canonical_project["train_split"],
            canonical_project["val_split"],
        )


def test_labels_clean_content_tampering_is_rejected(
    canonical_project: Dict[str, Path],
) -> None:
    with (canonical_project["label_dir"] / "train_jpg.txt").open(
        "a", encoding="utf-8", newline="\n"
    ) as stream:
        stream.write("1 0.5 0.5 0.2 0.2\n")
    with pytest.raises(MODULE.IRDatasetViewError, match="labels_clean identity 不一致"):
        MODULE.validate_sources(
            canonical_project["ir_dir"],
            canonical_project["label_dir"],
            canonical_project["train_split"],
            canonical_project["val_split"],
            expected_train_count=2,
            expected_val_count=1,
        )


def test_split_count_validation_is_strict(tmp_path: Path) -> None:
    split = tmp_path / "split.txt"
    write_text_lf(split, "only_one\n")
    with pytest.raises(MODULE.IRDatasetViewError, match="Split 数量错误"):
        MODULE.read_stems(split, expected_count=2)


def test_train_val_overlap_is_rejected(
    canonical_project: Dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    write_text_lf(canonical_project["val_split"], "train_jpg\n")
    configure_contract_hashes(monkeypatch, canonical_project)
    with pytest.raises(MODULE.IRDatasetViewError, match="train/val split 存在重复 stem"):
        MODULE.validate_sources(
            canonical_project["ir_dir"],
            canonical_project["label_dir"],
            canonical_project["train_split"],
            canonical_project["val_split"],
            expected_train_count=2,
            expected_val_count=1,
        )


@pytest.mark.parametrize("stems", [["same", "same"], ["ABC", "abc"]])
def test_exact_and_casefold_split_duplicates_are_rejected(
    tmp_path: Path, stems: List[str]
) -> None:
    split = tmp_path / "split.txt"
    write_text_lf(split, "\n".join(stems) + "\n")
    with pytest.raises(MODULE.IRDatasetViewError, match="重复或仅大小写不同"):
        MODULE.read_stems(split, expected_count=2)


class FakeFile:
    def __init__(self, name: str) -> None:
        self.name = name
        self.stem = Path(name).stem
        self.suffix = Path(name).suffix

    def is_file(self) -> bool:
        return True


class FakeDirectory:
    def __init__(self, names: List[str]) -> None:
        self.files = [FakeFile(name) for name in names]

    def is_dir(self) -> bool:
        return True

    def iterdir(self) -> List[FakeFile]:
        return self.files


@pytest.mark.parametrize(
    ("names", "extensions", "kind"),
    [
        (["ABC.png", "abc.jpg"], MODULE.IR_IMAGE_EXTENSIONS, "IR 图像"),
        (["ABC.txt", "abc.txt"], {".txt"}, "labels_clean"),
    ],
)
def test_casefold_file_collision_is_rejected_portably(
    names: List[str], extensions: set, kind: str
) -> None:
    directory = FakeDirectory(names)
    with pytest.raises(MODULE.IRDatasetViewError, match="casefold stem"):
        MODULE._index_unique_files(directory, extensions, kind)


def test_missing_image_is_rejected_before_output(
    canonical_project: Dict[str, Path],
) -> None:
    (canonical_project["ir_dir"] / "val_empty.png").unlink()
    with pytest.raises(MODULE.IRDatasetViewError, match="IR 图像.*缺失"):
        build_canonical(canonical_project)
    assert not canonical_project["output"].exists()


def test_missing_label_is_rejected_before_output(
    canonical_project: Dict[str, Path],
) -> None:
    (canonical_project["label_dir"] / "val_empty.txt").unlink()
    with pytest.raises(MODULE.IRDatasetViewError, match="labels_clean.*缺失"):
        build_canonical(canonical_project)
    assert not canonical_project["output"].exists()


@pytest.mark.parametrize(("first", "second"), [("same.png", "same.jpg"), ("ABC.png", "abc.jpg")])
def test_duplicate_and_casefold_ir_image_stems_are_rejected(
    canonical_project: Dict[str, Path],
    first: str,
    second: str,
) -> None:
    write_ir(canonical_project["ir_dir"] / first)
    write_ir(canonical_project["ir_dir"] / second)
    with pytest.raises(MODULE.IRDatasetViewError, match="casefold stem"):
        MODULE.index_ir_images(canonical_project["ir_dir"])


@pytest.mark.parametrize("extra_kind", ["image", "label"])
def test_split_external_source_stem_is_rejected(
    canonical_project: Dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    extra_kind: str,
) -> None:
    if extra_kind == "image":
        write_ir(canonical_project["ir_dir"] / "extra.png")
    else:
        write_text_lf(canonical_project["label_dir"] / "extra.txt", "")
        configure_contract_hashes(monkeypatch, canonical_project)
    with pytest.raises(MODULE.IRDatasetViewError, match="split 外额外 stem"):
        MODULE.validate_sources(
            canonical_project["ir_dir"],
            canonical_project["label_dir"],
            canonical_project["train_split"],
            canonical_project["val_split"],
            expected_train_count=2,
            expected_val_count=1,
        )


def test_default_copy_is_byte_preserving_and_isolated(
    canonical_project: Dict[str, Path],
) -> None:
    source_before = digest_tree(canonical_project["root"] / "data/raw")
    labels_before = digest_tree(canonical_project["label_dir"])
    result = build_canonical(canonical_project)
    manifest = result["manifest"]

    assert manifest["link_mode_requested"] == "copy"
    assert manifest["link_mode_effective"] == "copy"
    assert manifest["isolation"] is True
    for record in manifest["source_file_list"]:
        source_image = canonical_project["root"] / record["source_image_project_relative"]
        staged_image = canonical_project["output"] / record["staged_image_view_relative"]
        source_label = canonical_project["root"] / record["source_label_project_relative"]
        staged_label = canonical_project["output"] / record["staged_label_view_relative"]
        assert source_image.read_bytes() == staged_image.read_bytes()
        assert source_label.read_bytes() == staged_label.read_bytes()
        assert MODULE.sha256_file(source_image) == MODULE.sha256_file(staged_image)
        assert not os.path.samefile(source_image, staged_image)
        assert not os.path.samefile(source_label, staged_label)
    assert digest_tree(canonical_project["root"] / "data/raw") == source_before
    assert digest_tree(canonical_project["label_dir"]) == labels_before


def test_explicit_hardlink_is_nonisolated(
    canonical_project: Dict[str, Path],
) -> None:
    result = build_canonical(canonical_project, link_mode="hardlink")
    record = result["manifest"]["source_file_list"][0]
    source = canonical_project["root"] / record["source_image_project_relative"]
    staged = canonical_project["output"] / record["staged_image_view_relative"]
    assert os.path.samefile(source, staged)
    assert result["manifest"]["link_mode_requested"] == "hardlink"
    assert result["manifest"]["link_mode_effective"] == "hardlink"
    assert result["manifest"]["isolation"] is False


def test_auto_falls_back_to_copy(
    canonical_project: Dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_link(source: Path, destination: Path) -> None:
        raise OSError("hardlink unavailable")

    monkeypatch.setattr(MODULE.os, "link", fail_link)
    result = build_canonical(canonical_project, link_mode="auto")
    assert result["manifest"]["link_mode_effective"] == "copy"
    assert result["manifest"]["isolation"] is True


def test_existing_output_is_rejected_without_force(
    canonical_project: Dict[str, Path],
) -> None:
    build_canonical(canonical_project)
    marker = canonical_project["output"] / "keep.txt"
    write_text_lf(marker, "keep")
    with pytest.raises(MODULE.IRDatasetViewError, match="显式使用 --force"):
        build_canonical(canonical_project)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_force_rebuilds_only_raw3_and_preserves_sibling_and_sources(
    canonical_project: Dict[str, Path],
) -> None:
    build_canonical(canonical_project)
    sibling = canonical_project["output"].parent / "sentinel_other_view"
    sentinel = sibling / "keep.txt"
    write_text_lf(sentinel, "unchanged")
    write_text_lf(canonical_project["output"] / "obsolete.txt", "remove")
    raw_before = digest_tree(canonical_project["root"] / "data/raw")
    labels_before = digest_tree(canonical_project["label_dir"])
    splits_before = (
        canonical_project["train_split"].read_bytes(),
        canonical_project["val_split"].read_bytes(),
    )

    result = build_canonical(canonical_project, force=True)

    assert result["manifest"]["isolation"] is True
    assert not (canonical_project["output"] / "obsolete.txt").exists()
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert digest_tree(canonical_project["root"] / "data/raw") == raw_before
    assert digest_tree(canonical_project["label_dir"]) == labels_before
    assert canonical_project["train_split"].read_bytes() == splits_before[0]
    assert canonical_project["val_split"].read_bytes() == splits_before[1]


@pytest.mark.parametrize(
    "candidate_name",
    [
        "ir_equal",
        "ir_parent",
        "ir_child",
        "labels_equal",
        "labels_parent",
        "labels_child",
        "raw_root",
        "project_root",
        "ir_trainable_root",
    ],
)
def test_output_source_and_protected_path_overlap_is_rejected(
    canonical_project: Dict[str, Path],
    candidate_name: str,
) -> None:
    candidates = {
        "ir_equal": canonical_project["ir_dir"],
        "ir_parent": canonical_project["ir_dir"].parent,
        "ir_child": canonical_project["ir_dir"] / "subdir",
        "labels_equal": canonical_project["label_dir"],
        "labels_parent": canonical_project["label_dir"].parent,
        "labels_child": canonical_project["label_dir"] / "subdir",
        "raw_root": canonical_project["root"] / "data/raw",
        "project_root": canonical_project["root"],
        "ir_trainable_root": canonical_project["output"].parent,
    }
    with pytest.raises(MODULE.IRDatasetViewError, match="输出目录"):
        MODULE.validate_output_path(
            candidates[candidate_name],
            canonical_project["ir_dir"],
            canonical_project["label_dir"],
            canonical_project["train_split"],
            canonical_project["val_split"],
        )


def test_output_dotdot_normalizes_to_canonical_view(
    canonical_project: Dict[str, Path],
) -> None:
    candidate = canonical_project["output"].parent / "temporary" / ".." / "raw3"
    MODULE.validate_output_path(
        candidate,
        canonical_project["ir_dir"],
        canonical_project["label_dir"],
        canonical_project["train_split"],
        canonical_project["val_split"],
    )


def test_symlinked_canonical_output_to_source_is_rejected_when_supported(
    canonical_project: Dict[str, Path],
) -> None:
    try:
        os.symlink(
            canonical_project["ir_dir"],
            canonical_project["output"],
            target_is_directory=True,
        )
    except OSError:
        pytest.skip("Symlink creation is unavailable; normalization is covered separately")
    with pytest.raises(MODULE.IRDatasetViewError, match="重叠|resolve|symlink"):
        MODULE.validate_output_path(
            canonical_project["output"],
            canonical_project["ir_dir"],
            canonical_project["label_dir"],
            canonical_project["train_split"],
            canonical_project["val_split"],
        )


def test_manifest_is_byte_deterministic_and_machine_independent(
    canonical_project: Dict[str, Path],
) -> None:
    first = build_canonical(canonical_project)
    first_bytes = (canonical_project["output"] / "manifest.json").read_bytes()
    second = build_canonical(canonical_project, force=True)
    second_bytes = (canonical_project["output"] / "manifest.json").read_bytes()

    assert first_bytes == second_bytes
    assert first["manifest"] == second["manifest"]
    assert b"created_at" not in first_bytes
    parsed = json.loads(first_bytes)
    for key in ("source_ir_dir", "label_dir", "train_split_path", "val_split_path"):
        assert not Path(parsed[key]).is_absolute()
        assert "\\" not in parsed[key]
    for record in parsed["source_file_list"]:
        assert set(
            (
                "stem",
                "subset",
                "source_image_project_relative",
                "staged_image_view_relative",
                "source_label_project_relative",
                "staged_label_view_relative",
                "image_bytes",
                "image_sha256",
                "label_bytes",
                "label_sha256",
            )
        ).issubset(record)
        assert not Path(record["source_image_project_relative"]).is_absolute()
        assert not Path(record["staged_image_view_relative"]).is_absolute()


def test_data_yaml_has_standard_paths_and_canonical_names(
    canonical_project: Dict[str, Path],
) -> None:
    build_canonical(canonical_project)
    config = yaml.safe_load((canonical_project["output"] / "data.yaml").read_text(encoding="utf-8"))
    assert Path(config["path"]) == canonical_project["output"].resolve()
    assert config["train"] == "images/train"
    assert config["val"] == "images/val"
    assert tuple(config["names"].values()) == MODULE.CLASS_NAMES


@pytest.mark.parametrize("representation", ["gray3", "gray1", "normalize3", "clahe3"])
def test_non_raw3_representation_is_explicitly_rejected(
    canonical_project: Dict[str, Path],
    representation: str,
) -> None:
    with pytest.raises(MODULE.IRDatasetViewError, match="仅实现 representation=raw3"):
        MODULE.build_view(
            canonical_project["ir_dir"],
            canonical_project["label_dir"],
            canonical_project["train_split"],
            canonical_project["val_split"],
            canonical_project["output"],
            representation=representation,
            expected_train_count=2,
            expected_val_count=1,
        )


@pytest.mark.parametrize(
    ("dtype", "channels", "message"),
    [(np.uint16, 3, "uint8"), (np.uint8, 1, "3-channel")],
)
def test_raw3_encoding_contract_is_enforced(
    canonical_project: Dict[str, Path],
    dtype: np.dtype,
    channels: int,
    message: str,
) -> None:
    write_ir(canonical_project["ir_dir"] / "train_jpg.png", dtype=dtype, channels=channels)
    (canonical_project["ir_dir"] / "train_jpg.jpg").unlink()
    with pytest.raises(MODULE.IRDatasetViewError, match=message):
        MODULE.validate_sources(
            canonical_project["ir_dir"],
            canonical_project["label_dir"],
            canonical_project["train_split"],
            canonical_project["val_split"],
            expected_train_count=2,
            expected_val_count=1,
        )


def test_cli_defaults_to_canonical_inputs_raw3_and_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])
    args = MODULE.parse_args()
    assert args.ir_dir == "data/raw/train/infrared"
    assert args.label_dir == "data/processed/train/labels_clean"
    assert args.train_split == "data/splits/train.txt"
    assert args.val_split == "data/splits/val.txt"
    assert args.output_root == "data/processed/ir_trainable/raw3"
    assert args.representation == "raw3"
    assert args.link_mode == "copy"


def test_cli_help_marks_hardlink_as_nonisolated(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--help"])
    with pytest.raises(SystemExit) as exc_info:
        MODULE.parse_args()
    assert exc_info.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "shares file identity/inode with source" in help_text
    assert "non-isolated" in help_text


def test_python38_parse_and_no_path_write_text_usage() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    ast.parse(source, filename=str(SCRIPT), feature_version=(3, 8))
    assert ".write_text(" not in source
