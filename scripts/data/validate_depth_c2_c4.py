"""Reproduce C2-C4 Depth staging and Ultralytics loader validation evidence."""

import argparse
import ast
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import cv2
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prepare_depth_yolo as c2  # noqa: E402


CANDIDATES = ("validmask", "percentile", "log", "inverse")
STAGING_ROOT = PROJECT_ROOT / "data" / "processed" / "depth_trainable"
LABELS_CLEAN = PROJECT_ROOT / c2.CANONICAL_LABEL_RELATIVE
TRAIN_SPLIT = PROJECT_ROOT / c2.CANONICAL_TRAIN_SPLIT_RELATIVE
VAL_SPLIT = PROJECT_ROOT / c2.CANONICAL_VAL_SPLIT_RELATIVE
JSON_OUTPUT = PROJECT_ROOT / "outputs" / "analysis" / "depth_c2_c4_validation.json"
LOG_OUTPUT = PROJECT_ROOT / "outputs" / "analysis" / "depth_c2_c4_validation.log"
COMMAND = "python scripts/data/validate_depth_c2_c4.py"
EXPECTED_PERCENTILE_PARAMETER_SHA256 = (
    "aab4008fb51154016d7703de6af186e39267c322b71e93e791aa28e1b646f6ab"
)


class DepthValidationError(RuntimeError):
    """Raised when reproducible staging evidence fails its frozen contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DepthValidationError(message)


def git_head_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(PROJECT_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    require(len(value) == 40 and all(char in "0123456789abcdef" for char in value), "Git HEAD 不是完整 SHA")
    return value


def index_files(directory: Path, extensions: Set[str]) -> Tuple[Dict[str, List[Path]], int]:
    require(directory.is_dir(), "目录不存在: {}".format(directory))
    index: Dict[str, List[Path]] = {}
    count = 0
    for path in sorted(directory.iterdir(), key=lambda item: (item.name.casefold(), item.name)):
        if path.is_file() and path.suffix.lower() in extensions:
            index.setdefault(path.stem.casefold(), []).append(path)
            count += 1
    return index, count


def duplicate_count(index: Dict[str, List[Path]]) -> int:
    return sum(max(0, len(paths) - 1) for paths in index.values())


def python38_static_compatibility() -> str:
    ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"), filename=str(SCRIPT_PATH), feature_version=(3, 8))
    return "passed"


def collect_source_contract() -> Tuple[Dict[str, Any], List[str], List[str]]:
    train_stems = c2.read_stems(TRAIN_SPLIT, c2.EXPECTED_TRAIN_COUNT)
    val_stems = c2.read_stems(VAL_SPLIT, c2.EXPECTED_VAL_COUNT)
    train_keys = {stem.casefold() for stem in train_stems}
    val_keys = {stem.casefold() for stem in val_stems}
    overlap = train_keys & val_keys
    require(not overlap, "fixed split 存在 train/val overlap")

    train_sha = c2.normalized_split_sha256(TRAIN_SPLIT)
    val_sha = c2.normalized_split_sha256(VAL_SPLIT)
    require(train_sha == c2.CANONICAL_TRAIN_NORMALIZED_SHA256, "train split normalized SHA256 不一致")
    require(val_sha == c2.CANONICAL_VAL_NORMALIZED_SHA256, "val split normalized SHA256 不一致")

    labels = c2.index_labels(LABELS_CLEAN)
    labels_identity, _ = c2._aggregate_paths(list(labels.values()))
    require(len(labels) == c2.EXPECTED_TRAIN_COUNT + c2.EXPECTED_VAL_COUNT, "labels_clean 数量不一致")
    require(
        labels_identity["aggregate_sha256"] == c2.CANONICAL_LABELS_CLEAN_SHA256,
        "labels_clean aggregate SHA256 不一致",
    )
    require({stem.casefold() for stem in labels} == train_keys | val_keys, "labels_clean stem 与 fixed split 不一致")
    return (
        {
            "train_split_path": c2.CANONICAL_TRAIN_SPLIT_RELATIVE.as_posix(),
            "val_split_path": c2.CANONICAL_VAL_SPLIT_RELATIVE.as_posix(),
            "train_split_normalized_sha256": train_sha,
            "val_split_normalized_sha256": val_sha,
            "train_count": len(train_stems),
            "val_count": len(val_stems),
            "train_val_overlap_count": len(overlap),
            "labels_clean_path": c2.CANONICAL_LABEL_RELATIVE.as_posix(),
            "labels_clean_count": labels_identity["file_count"],
            "labels_clean_total_bytes": labels_identity["total_bytes"],
            "labels_clean_sha256": labels_identity["aggregate_sha256"],
        },
        train_stems,
        val_stems,
    )


def representation_definition(candidate: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
    require(manifest.get("candidate") == candidate, "{} manifest candidate 不一致".format(candidate))
    require(manifest.get("representation") == candidate, "{} manifest representation 不一致".format(candidate))
    conversion = dict(manifest.get("conversion", {}))
    require(conversion.get("output_dtype") == "uint8", "{} output_dtype 不一致".format(candidate))
    require(conversion.get("output_channels") == 3, "{} channels 不一致".format(candidate))

    if candidate == "validmask":
        require(conversion.get("valid_rule") == "depth_mm > 0", "validmask valid_rule 不一致")
        require(conversion.get("invalid_value") == 0, "validmask invalid_value 不一致")
        require(conversion.get("valid_value") == 255, "validmask valid_value 不一致")
    elif candidate == "percentile":
        require(conversion.get("p_low") == 2 and conversion.get("p_high") == 98, "percentile P2/P98 不一致")
        require(conversion.get("low_mm") == 1638 and conversion.get("high_mm") == 18819, "percentile mm 参数不一致")
        require(
            conversion.get("c3_parameter_identity_sha256") == EXPECTED_PERCENTILE_PARAMETER_SHA256,
            "percentile C3 参数 SHA256 不一致",
        )
        conversion.update(
            {
                "fit_scope": "train_png_only",
                "zero_excluded": True,
                "val_used_for_fit": False,
                "jpg_used_for_fit": False,
            }
        )
    elif candidate == "log":
        require(conversion.get("formula") == "log1p(depth_mm)", "log formula 不一致")
        require(conversion.get("model_range_min_mm") == 1, "log fixed min 不一致")
        require(conversion.get("model_range_max_mm") == 19999, "log fixed max 不一致")
    elif candidate == "inverse":
        require(conversion.get("near_clip_mm") == 300, "inverse near clip 不一致")
        require(conversion.get("far_clip_mm") == 19999, "inverse far clip 不一致")
        require(conversion.get("invalid_value") == 0, "inverse invalid value 不一致")
        require(conversion.get("valid_output_range") == [1, 255], "inverse output range 不一致")
    return conversion


def validate_staging_candidate(
    candidate: str, train_stems: Sequence[str], val_stems: Sequence[str], source_contract: Dict[str, Any]
) -> Dict[str, Any]:
    view = STAGING_ROOT / candidate
    manifest_path = view / "manifest.json"
    require(manifest_path.is_file(), "{} manifest 不存在".format(candidate))
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    definition = representation_definition(candidate, manifest)
    train_keys = {stem.casefold() for stem in train_stems}
    val_keys = {stem.casefold() for stem in val_stems}

    subset_indexes: Dict[str, Dict[str, List[Path]]] = {}
    subset_label_indexes: Dict[str, Dict[str, List[Path]]] = {}
    image_counts: Dict[str, int] = {}
    label_counts: Dict[str, int] = {}
    missing_images = 0
    missing_labels = 0
    extra_images = 0
    extra_labels = 0
    for subset, expected_keys in (("train", train_keys), ("val", val_keys)):
        image_index, image_count = index_files(view / "images" / subset, c2.DEPTH_IMAGE_EXTENSIONS)
        label_index, label_count = index_files(view / "labels" / subset, {".txt"})
        subset_indexes[subset] = image_index
        subset_label_indexes[subset] = label_index
        image_counts[subset] = image_count
        label_counts[subset] = label_count
        actual_image_keys = set(image_index)
        actual_label_keys = set(label_index)
        missing_images += len(expected_keys - actual_image_keys)
        missing_labels += len(expected_keys - actual_label_keys)
        extra_images += len(actual_image_keys - expected_keys)
        extra_labels += len(actual_label_keys - expected_keys)

    duplicate_images = sum(duplicate_count(index) for index in subset_indexes.values())
    duplicate_labels = sum(duplicate_count(index) for index in subset_label_indexes.values())
    actual_train_keys = set(subset_indexes["train"]) | set(subset_label_indexes["train"])
    actual_val_keys = set(subset_indexes["val"]) | set(subset_label_indexes["val"])
    overlap_count = len(actual_train_keys & actual_val_keys)

    decoded = 0
    decode_failures = 0
    observed_dtypes: Set[str] = set()
    observed_channels: Set[int] = set()
    for subset in ("train", "val"):
        for paths in subset_indexes[subset].values():
            for path in paths:
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    decode_failures += 1
                    continue
                decoded += 1
                observed_dtypes.add(str(image.dtype))
                observed_channels.add(1 if image.ndim == 2 else int(image.shape[2]))

    manifest_pairs = [
        (str(record.get("subset")), str(record.get("stem")).casefold())
        for record in manifest.get("source_file_list", [])
    ]
    expected_pairs = [("train", stem.casefold()) for stem in train_stems] + [
        ("val", stem.casefold()) for stem in val_stems
    ]
    manifest_pair_duplicates = len(manifest_pairs) - len(set(manifest_pairs))
    require(set(manifest_pairs) == set(expected_pairs), "{} manifest sample index 与 fixed split 不一致".format(candidate))
    require(manifest_pair_duplicates == 0, "{} manifest sample index 包含重复".format(candidate))
    require(manifest.get("train_split_normalized_sha256") == source_contract["train_split_normalized_sha256"], "{} manifest train split SHA 不一致".format(candidate))
    require(manifest.get("val_split_normalized_sha256") == source_contract["val_split_normalized_sha256"], "{} manifest val split SHA 不一致".format(candidate))
    require(manifest.get("labels_clean_identity", {}).get("aggregate_sha256") == source_contract["labels_clean_sha256"], "{} manifest labels identity 不一致".format(candidate))

    total = image_counts["train"] + image_counts["val"]
    missing_count = missing_images + missing_labels
    duplicate_total = duplicate_images + duplicate_labels
    require(total == 2000 and image_counts == {"train": 1600, "val": 400}, "{} 实际 image 数量不一致".format(candidate))
    require(label_counts == {"train": 1600, "val": 400}, "{} 实际 label 数量不一致".format(candidate))
    require(missing_count == 0 and extra_images == 0 and extra_labels == 0, "{} image/label 一一对应失败".format(candidate))
    require(duplicate_total == 0 and overlap_count == 0, "{} duplicate/overlap 不为 0".format(candidate))
    require(decoded == 2000 and decode_failures == 0, "{} output decode 失败".format(candidate))
    require(observed_dtypes == {"uint8"}, "{} observed dtype 不符合 uint8".format(candidate))
    require(observed_channels == {3}, "{} observed channels 不符合 3".format(candidate))

    return {
        "candidate": candidate,
        "representation": manifest["representation"],
        "representation_definition": definition,
        "manifest_path": (Path("data") / "processed" / "depth_trainable" / candidate / "manifest.json").as_posix(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "total": total,
        "train": image_counts["train"],
        "val": image_counts["val"],
        "train_label_count": label_counts["train"],
        "val_label_count": label_counts["val"],
        "missing_image_count": missing_images,
        "missing_label_count": missing_labels,
        "missing_count": missing_count,
        "extra_image_count": extra_images,
        "extra_label_count": extra_labels,
        "duplicate_image_count": duplicate_images,
        "duplicate_label_count": duplicate_labels,
        "duplicate_count": duplicate_total,
        "train_val_overlap_count": overlap_count,
        "decoded_image_count": decoded,
        "decode_failure_count": decode_failures,
        "observed_dtypes": sorted(observed_dtypes),
        "observed_channels": sorted(observed_channels),
        "jpg_policy": manifest.get("jpg_policy"),
    }


def summarize_batch(batch: Dict[str, Any], dataset_size: int) -> Dict[str, Any]:
    import torch

    require("img" in batch, "Ultralytics batch 缺少 img")
    image = batch["img"]
    require(isinstance(image, torch.Tensor) and image.ndim == 4, "Ultralytics img tensor 非标准 NCHW")
    cls = batch.get("cls")
    bboxes = batch.get("bboxes")
    require(isinstance(cls, torch.Tensor), "Ultralytics batch 缺少 cls tensor")
    require(isinstance(bboxes, torch.Tensor), "Ultralytics batch 缺少 bboxes tensor")
    require(torch.isfinite(image.float()).all().item(), "image tensor 包含非有限值")
    require(torch.isfinite(cls.float()).all().item(), "class tensor 包含非有限值")
    require(torch.isfinite(bboxes.float()).all().item(), "bbox tensor 包含非有限值")
    return {
        "batch_obtained": True,
        "dataset_size": int(dataset_size),
        "image_shape": [int(value) for value in image.shape],
        "image_dtype": str(image.dtype),
        "configured_batch_size": 2,
        "actual_sample_count": int(image.shape[0]),
        "finite": True,
        "image_min": float(image.min().item()),
        "image_max": float(image.max().item()),
        "preprocessing_stage": "Ultralytics dataset/dataloader output before trainer normalization",
        "class_tensor_present": True,
        "class_tensor_shape": [int(value) for value in cls.shape],
        "classes_read": True,
        "bbox_tensor_present": True,
        "bbox_tensor_shape": [int(value) for value in bboxes.shape],
        "bboxes_read": True,
    }


def loader_smoke(candidate: str, mode: str, expected_size: int) -> Dict[str, Any]:
    os.environ["YOLO_CONFIG_DIR"] = str(STAGING_ROOT)
    os.environ.setdefault("MPLCONFIGDIR", str(STAGING_ROOT / "Matplotlib"))
    import torch
    from ultralytics.cfg import DEFAULT_CFG, get_cfg
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    data = check_det_dataset(str(STAGING_ROOT / candidate / "data.yaml"))
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = 640
    cfg.cache = False
    cfg.rect = False
    cfg.workers = 0
    cfg.task = "detect"
    dataset = build_yolo_dataset(
        cfg,
        img_path=data[mode],
        batch=2,
        data=data,
        mode=mode,
        rect=False,
        stride=32,
    )
    require(len(dataset) == expected_size, "{} {} dataset size 不一致".format(candidate, mode))
    loader = build_dataloader(dataset, batch=2, workers=0, shuffle=False, device="cpu")
    batch = next(iter(loader))
    return summarize_batch(batch, len(dataset))


def environment_info() -> Dict[str, Any]:
    os.environ["YOLO_CONFIG_DIR"] = str(STAGING_ROOT)
    import torch
    import ultralytics

    return {
        "python_version": platform.python_version(),
        "python_runtime": sys.version,
        "pytorch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "python38_static_compatibility": python38_static_compatibility(),
    }


def render_log(payload: Dict[str, Any]) -> str:
    environment = payload["environment"]
    lines = [
        "Command: {}".format(payload["command"]),
        "Git HEAD: {}".format(payload["git_head_sha"]),
        "Python version: {}".format(environment["python_version"]),
        "PyTorch version: {}".format(environment["pytorch_version"]),
        "Ultralytics version: {}".format(environment["ultralytics_version"]),
        "Python 3.8 static compatibility: {}".format(environment["python38_static_compatibility"].upper()),
        "",
    ]
    for candidate in payload.get("candidates", []):
        lines.append("Candidate: {}".format(candidate["candidate"]))
        lines.append("Manifest SHA256: {}".format(candidate["manifest_sha256"]))
        for mode in ("train", "val"):
            smoke = candidate["{}_loader_smoke".format(mode)]
            lines.append(
                "{}: dataset_size={} first_batch={} shape={} dtype={} actual_samples={} cls_shape={} bbox_shape={} PASS".format(
                    mode,
                    smoke["dataset_size"],
                    smoke["batch_obtained"],
                    smoke["image_shape"],
                    smoke["image_dtype"],
                    smoke["actual_sample_count"],
                    smoke["class_tensor_shape"],
                    smoke["bbox_tensor_shape"],
                )
            )
        lines.append("")
    lines.append("OVERALL: {}".format(payload["overall_status"]))
    return "\n".join(lines) + "\n"


def write_outputs(payload: Dict[str, Any]) -> None:
    JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    log_text = render_log(payload)
    json_temp = JSON_OUTPUT.with_suffix(".json.tmp")
    log_temp = LOG_OUTPUT.with_suffix(".log.tmp")
    json_temp.write_text(json_text, encoding="utf-8", newline="\n")
    log_temp.write_text(log_text, encoding="utf-8", newline="\n")
    json_temp.replace(JSON_OUTPUT)
    log_temp.replace(LOG_OUTPUT)


def validate() -> Dict[str, Any]:
    head = git_head_sha()
    source_contract, train_stems, val_stems = collect_source_contract()
    candidates = []
    for candidate in CANDIDATES:
        result = validate_staging_candidate(candidate, train_stems, val_stems, source_contract)
        result["train_loader_smoke"] = loader_smoke(candidate, "train", 1600)
        result["val_loader_smoke"] = loader_smoke(candidate, "val", 400)
        candidates.append(result)
    return {
        "schema_version": 2,
        "command": COMMAND,
        "git_head_sha": head,
        "environment": environment_info(),
        "source_contract": source_contract,
        "candidates": candidates,
        "overall_status": "PASS",
    }


def parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv: Sequence[str] = None) -> int:
    parse_args(argv)
    try:
        payload = validate()
    except Exception as exc:
        failure = {
            "schema_version": 2,
            "command": COMMAND,
            "git_head_sha": git_head_sha(),
            "environment": environment_info(),
            "candidates": [],
            "overall_status": "FAIL",
            "error": "{}: {}".format(type(exc).__name__, exc),
        }
        write_outputs(failure)
        print("OVERALL: FAIL - {}".format(exc), file=sys.stderr)
        return 1
    write_outputs(payload)
    print(render_log(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
