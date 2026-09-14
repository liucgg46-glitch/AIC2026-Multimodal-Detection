"""Run a configuration-driven Ultralytics RGB training experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
import ultralytics
import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RGB_CLEAN_DATA = Path("data/processed/rgb_yolo_clean/data.yaml")
CANONICAL_LABELS = Path("data/processed/train/labels_clean")
CANONICAL_LABELS_CLEAN_SHA256 = "6a670b95b33e803e5d25fc30d7bbd7985cbc4799234c37ff42b3b1c9204025a4"
RGB_CLEAN_CONTRACT = "rgb_labels_clean_v1"
CANONICAL_TRAIN_COUNT = 1600
CANONICAL_VAL_COUNT = 400


class TrainingConfigError(ValueError):
    """Raised when an experiment configuration cannot be used safely."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_labels(paths: List[Path]) -> Dict[str, object]:
    digest = hashlib.sha256()
    total_bytes = 0
    for path in sorted(paths, key=lambda item: (item.name.casefold(), item.name)):
        size = path.stat().st_size
        digest.update(f"{path.name}\0{size}\0{sha256(path)}\n".encode("utf-8"))
        total_bytes += size
    return {"aggregate_sha256": digest.hexdigest(), "file_count": len(paths), "total_bytes": total_bytes}


def validate_clean_rgb_view(data_path: Path) -> None:
    expected = (PROJECT_ROOT / RGB_CLEAN_DATA).resolve()
    if data_path.resolve() != expected:
        raise TrainingConfigError(f"clean RGB contract 要求 data: {RGB_CLEAN_DATA.as_posix()}")
    manifest_path = data_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise TrainingConfigError("clean RGB view 缺少 manifest.json；请重新生成")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingConfigError("clean RGB manifest.json 无法读取或格式损坏") from exc
    if (manifest.get("representation") != "rgb_byte_preserving_clean_labels"
            or manifest.get("train_count") != CANONICAL_TRAIN_COUNT
            or manifest.get("val_count") != CANONICAL_VAL_COUNT):
        raise TrainingConfigError("clean RGB manifest contract 不一致")

    train_stems = [line.strip() for line in (PROJECT_ROOT / "data/splits/train.txt").read_text(
        encoding="utf-8-sig").splitlines() if line.strip()]
    val_stems = [line.strip() for line in (PROJECT_ROOT / "data/splits/val.txt").read_text(
        encoding="utf-8-sig").splitlines() if line.strip()]
    if (len(train_stems) != CANONICAL_TRAIN_COUNT
            or len(val_stems) != CANONICAL_VAL_COUNT
            or set(train_stems) & set(val_stems)):
        raise TrainingConfigError("clean RGB contract 要求固定且无重叠的 1600/400 split")
    canonical = [PROJECT_ROOT / CANONICAL_LABELS / (stem + ".txt") for stem in train_stems + val_stems]
    staged = ([data_path.parent / "labels/train" / (stem + ".txt") for stem in train_stems]
              + [data_path.parent / "labels/val" / (stem + ".txt") for stem in val_stems])
    if any(not path.is_file() for path in canonical + staged):
        raise TrainingConfigError("clean RGB view 或 canonical labels_clean 文件缺失")
    canonical_identity = aggregate_labels(canonical)
    staged_identity = aggregate_labels(staged)
    if canonical_identity["aggregate_sha256"] != CANONICAL_LABELS_CLEAN_SHA256:
        raise TrainingConfigError("canonical labels_clean SHA256 不一致")
    if staged_identity != canonical_identity or manifest.get("labels_identity") != canonical_identity:
        raise TrainingConfigError("clean RGB view 标签与 canonical labels_clean 不一致")


def project_path(value: Union[str, Path]) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise TrainingConfigError(f"实验配置不存在: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TrainingConfigError(f"实验配置必须是 YAML mapping: {path}")

    missing = [key for key in ("experiment_id", "model", "data") if not config.get(key)]
    if missing:
        raise TrainingConfigError(f"实验配置缺少必要字段: {missing}")

    data_contract = config.pop("data_contract", None)
    data_path = project_path(config["data"])
    if not data_path.is_file():
        raise TrainingConfigError(
            f"Ultralytics data 配置不存在: {data_path}\n"
            "请先运行 prepare_rgb_yolo.py 生成兼容视图。"
        )
    config["data"] = str(data_path)
    config["project"] = str(project_path(config.get("project", "runs")))

    model = Path(str(config["model"]))
    pretrained = config.get("pretrained", True)
    if model.suffix.lower() in {".yaml", ".yml"} and pretrained is True:
        raise TrainingConfigError(
            "model YAML + pretrained=true 不会自动加载 COCO 权重。"
            "请使用 model: weights/yolo11n.pt，或明确指定 pretrained 权重路径；"
            "从零训练请显式设置 pretrained: false。"
        )
    if data_contract is not None:
        if data_contract != RGB_CLEAN_CONTRACT:
            raise TrainingConfigError(f"未知 data_contract: {data_contract}")
        validate_clean_rgb_view(data_path)
    if isinstance(pretrained, str):
        pretrained_path = project_path(pretrained)
        if not pretrained_path.is_file():
            raise TrainingConfigError(f"预训练权重不存在: {pretrained_path}")
        config["pretrained"] = str(pretrained_path)
    local_model = project_path(model)
    if model.suffix.lower() == ".pt" and not local_model.is_file():
        raise TrainingConfigError(f"请先准备本地离线权重: {local_model}")
    if local_model.is_file():
        config["model"] = str(local_model)
    return config


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def require_formal_checkout(expected_sha: str) -> None:
    if len(expected_sha) != 40 or any(char not in "0123456789abcdefABCDEF" for char in expected_sha):
        raise TrainingConfigError("--expected-sha 必须是完整的 40 位 Git SHA")
    commit = git_commit()
    if commit.lower() != expected_sha.lower():
        raise TrainingConfigError(f"HEAD 与 --expected-sha 不一致: {commit} != {expected_sha}")
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=PROJECT_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if status:
        raise TrainingConfigError("正式训练要求 clean working tree")
    if branch:
        raise TrainingConfigError(f"正式训练要求 detached HEAD，当前分支: {branch}")


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/SMOKE_RGB_001.yaml",
        help="Experiment YAML path, relative to the repository root unless absolute.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="Run the frozen formal experiment")
    mode.add_argument("--smoke", action="store_true", help="Run one short GPU integration epoch")
    parser.add_argument("--expected-sha", default="")
    return parser.parse_args()


def main() -> int:
    configure_console_encoding()
    args = parse_args()
    try:
        config = load_config(project_path(args.config))
    except (OSError, TrainingConfigError, yaml.YAMLError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    experiment_id = str(config.pop("experiment_id"))
    if args.train:
        try:
            require_formal_checkout(args.expected_sha)
        except (OSError, subprocess.SubprocessError, TrainingConfigError) as exc:
            print(f"错误: {exc}", file=sys.stderr)
            return 2
    elif args.expected_sha:
        print("错误: --expected-sha 只用于 --train", file=sys.stderr)
        return 2

    mode_name = "formal" if args.train else "smoke" if args.smoke else "check-only"
    model_source = str(config.pop("model"))
    print(f"Experiment: {experiment_id}")
    print(f"Mode: {mode_name}")
    print(f"Git commit: {git_commit()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Ultralytics: {ultralytics.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Model: {model_source}")
    print(f"Data: {config['data']}")

    if not (args.train or args.smoke):
        print("配置与本地权重检查通过；未启动训练。")
        return 0

    if args.smoke:
        config.update(
            epochs=1,
            patience=1,
            batch=min(int(config.get("batch", 1)), 2),
            workers=0,
            imgsz=min(int(config.get("imgsz", 640)), 320),
            fraction=0.05,
            plots=False,
            name=experiment_id + "_SMOKE",
            exist_ok=True,
        )

    started = time.perf_counter()
    model = YOLO(model_source)
    metrics = model.train(**config)
    elapsed = time.perf_counter() - started

    save_dir = Path(model.trainer.save_dir).resolve()
    print(f"训练完成，耗时: {elapsed:.1f} 秒")
    print(f"Run directory: {save_dir}")
    print(f"Best checkpoint: {save_dir / 'weights' / 'best.pt'}")
    results = getattr(metrics, "results_dict", None)
    if results:
        print("Validation metrics:")
        for key, value in results.items():
            print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
