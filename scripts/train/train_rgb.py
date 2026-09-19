"""Run a configuration-driven Ultralytics single-modality detection experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RGB_CLEAN_DATA = Path("data/processed/rgb_yolo_clean/data.yaml")
CANONICAL_LABELS = Path("data/processed/train/labels_clean")
CANONICAL_LABELS_CLEAN_SHA256 = "6a670b95b33e803e5d25fc30d7bbd7985cbc4799234c37ff42b3b1c9204025a4"
RGB_CLEAN_CONTRACT = "rgb_labels_clean_v1"
IR_RAW3_CLEAN_CONTRACT = "ir_raw3_labels_clean_v1"
DEPTH_LOG_CLEAN_CONTRACT = "depth_log_labels_clean_v1"
CANONICAL_TRAIN_COUNT = 1600
CANONICAL_VAL_COUNT = 400
P2_INITIALIZATION_POLICY = "yolo11_p2_semantic_v1"
P2_LAYER_REMAP = {23: 17, 25: 19, 26: 20, 28: 22}


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


def p2_source_key(target_key: str) -> Union[str, None]:
    """Map a YOLO11-P2 target tensor to the semantically equivalent P3-P5 source tensor."""
    parts = target_key.split(".")
    if len(parts) < 3 or parts[0] != "model" or not parts[1].isdigit():
        return None
    layer = int(parts[1])
    suffix = ".".join(parts[2:])
    if layer <= 16:
        return target_key
    if layer in P2_LAYER_REMAP:
        return "model.%d.%s" % (P2_LAYER_REMAP[layer], suffix)
    if layer == 29 and suffix.startswith("cv2."):
        branch_parts = suffix.split(".")
        branch = int(branch_parts[1])
        if 1 <= branch <= 3:
            branch_parts[1] = str(branch - 1)
            return "model.23." + ".".join(branch_parts)
    if layer == 29 and suffix == "dfl.conv.weight":
        return "model.23.dfl.conv.weight"
    return None


def build_p2_initial_state(
    target: Dict[str, torch.Tensor], source: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    selected: Dict[str, torch.Tensor] = {}
    mappings: Dict[str, str] = {}
    mismatches = []
    for target_key, target_tensor in target.items():
        source_key = p2_source_key(target_key)
        if source_key is None or source_key not in source:
            continue
        if source[source_key].shape != target_tensor.shape:
            mismatches.append({
                "target": target_key,
                "source": source_key,
                "target_shape": list(target_tensor.shape),
                "source_shape": list(source[source_key].shape),
            })
            continue
        selected[target_key] = source[source_key]
        mappings[target_key] = source_key

    required_prefixes = tuple("model.%d." % layer for layer in range(17))
    missing_required = [
        key for key in target
        if key.startswith(required_prefixes) and key not in selected
    ]
    if missing_required:
        raise TrainingConfigError(
            "P2 初始化未完整迁移 backbone/top-down P3: " + str(missing_required[:5])
        )
    loaded_numel = sum(tensor.numel() for tensor in selected.values())
    target_numel = sum(tensor.numel() for tensor in target.values())
    loaded_numel_ratio = loaded_numel / target_numel
    report: Dict[str, object] = {
        "policy": P2_INITIALIZATION_POLICY,
        "loaded_tensors": len(selected),
        "target_tensors": len(target),
        "loaded_numel": loaded_numel,
        "target_numel": target_numel,
        "loaded_numel_ratio": loaded_numel_ratio,
        "mappings": mappings,
        "shape_mismatches": mismatches,
        "random_target_prefixes": ["model.19.", "model.20.", "model.22.", "model.29.cv3."],
    }
    if loaded_numel_ratio < 0.95:
        raise TrainingConfigError("P2 语义初始化参数覆盖率低于 95%")
    return selected, report


def initialize_p2_model(model: torch.nn.Module, checkpoint_path: Path) -> Dict[str, object]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    source_model = checkpoint.get("ema") or checkpoint.get("model")
    if source_model is None:
        raise TrainingConfigError("初始化 checkpoint 缺少 model/ema")
    selected, report = build_p2_initial_state(model.state_dict(), source_model.float().state_dict())
    model.load_state_dict(selected, strict=False)
    compact_report = {
        key: value for key, value in report.items() if key != "mappings"
    }
    print("P2_INITIALIZATION_REPORT=" + json.dumps(compact_report, sort_keys=True))
    return report


def build_initializing_trainer(checkpoint_path: Path, policy: str):
    """Initialize the model created by YOLO.train(), not its discarded YAML preview."""
    class InitializingDetectionTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            if weights is not None:
                raise TrainingConfigError("自定义 YAML 初始化不应同时收到 trainer weights")
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            if policy == P2_INITIALIZATION_POLICY:
                initialize_p2_model(model, checkpoint_path)
            elif policy == "ultralytics_shape_match":
                checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
                source_model = checkpoint.get("ema") or checkpoint.get("model")
                if source_model is None:
                    raise TrainingConfigError("初始化 checkpoint 缺少 model/ema")
                model.load(source_model)
                print("INITIAL_WEIGHTS_APPLIED_IN_TRAINER=" + str(checkpoint_path))
            else:
                raise TrainingConfigError("未知 initial_weights_policy: " + str(policy))
            return model

    return InitializingDetectionTrainer


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


def validate_clean_modality_view(data_path: Path, contract: str) -> None:
    views = {
        IR_RAW3_CLEAN_CONTRACT: (Path("data/processed/ir_trainable/raw3/data.yaml"), "raw3"),
        DEPTH_LOG_CLEAN_CONTRACT: (Path("data/processed/depth_trainable/log/data.yaml"), "log"),
    }
    expected_path, representation = views[contract]
    if data_path.resolve() != (PROJECT_ROOT / expected_path).resolve():
        raise TrainingConfigError(f"{contract} 要求 data: {expected_path.as_posix()}")
    manifest_path = data_path.parent / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingConfigError(f"{contract} 缺少或损坏 manifest.json") from exc
    if (manifest.get("representation") != representation
            or manifest.get("train_count") != CANONICAL_TRAIN_COUNT
            or manifest.get("val_count") != CANONICAL_VAL_COUNT
            or manifest.get("label_dir") != CANONICAL_LABELS.as_posix()):
        raise TrainingConfigError(f"{contract} manifest 数据来源或 split 不一致")
    if contract == DEPTH_LOG_CLEAN_CONTRACT and (
            manifest.get("train_jpg_count") != 122 or manifest.get("val_jpg_count") != 27
            or manifest.get("jpg_policy", {}).get("physical_unit") != "unknown"):
        raise TrainingConfigError("Depth log view 的 PNG/JPG 格式合同不一致")
    train_stems = [line.strip() for line in (PROJECT_ROOT / "data/splits/train.txt").read_text(
        encoding="utf-8-sig").splitlines() if line.strip()]
    val_stems = [line.strip() for line in (PROJECT_ROOT / "data/splits/val.txt").read_text(
        encoding="utf-8-sig").splitlines() if line.strip()]
    if (len(train_stems) != CANONICAL_TRAIN_COUNT or len(val_stems) != CANONICAL_VAL_COUNT
            or len(set(train_stems + val_stems)) != CANONICAL_TRAIN_COUNT + CANONICAL_VAL_COUNT):
        raise TrainingConfigError(f"{contract} 需要固定且无重叠的 1600/400 split")
    for subset, expected_stems in (("train", train_stems), ("val", val_stems)):
        image_dir = data_path.parent / "images" / subset
        actual_stems = [path.stem for path in image_dir.iterdir() if path.is_file()]
        if len(actual_stems) != len(expected_stems) or set(actual_stems) != set(expected_stems):
            raise TrainingConfigError(f"{contract} images/{subset} 与固定 split 不一致")
    canonical = [PROJECT_ROOT / CANONICAL_LABELS / (stem + ".txt") for stem in train_stems + val_stems]
    staged = ([data_path.parent / "labels/train" / (stem + ".txt") for stem in train_stems]
              + [data_path.parent / "labels/val" / (stem + ".txt") for stem in val_stems])
    if any(not path.is_file() for path in canonical + staged):
        raise TrainingConfigError(f"{contract} labels_clean 文件缺失")
    source_identity = aggregate_labels(canonical)
    if (source_identity["aggregate_sha256"] != CANONICAL_LABELS_CLEAN_SHA256
            or aggregate_labels(staged) != source_identity):
        raise TrainingConfigError(f"{contract} labels_clean 内容不一致")


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
    model_sha256 = config.pop("model_sha256", None)
    pretrained = config.get("pretrained", True)
    initial_weights = config.get("initial_weights")
    initial_weights_sha256 = config.pop("initial_weights_sha256", None)
    initial_weights_policy = config.get("initial_weights_policy")
    if model.suffix.lower() in {".yaml", ".yml"} and pretrained is True:
        raise TrainingConfigError(
            "model YAML + pretrained=true 不会自动加载 COCO 权重。"
            "请使用 model: weights/yolo11n.pt，或明确指定 pretrained 权重路径；"
            "从零训练请显式设置 pretrained: false。"
        )
    if data_contract is not None:
        if data_contract not in {RGB_CLEAN_CONTRACT, IR_RAW3_CLEAN_CONTRACT,
                                 DEPTH_LOG_CLEAN_CONTRACT}:
            raise TrainingConfigError(f"未知 data_contract: {data_contract}")
        if data_contract == RGB_CLEAN_CONTRACT:
            validate_clean_rgb_view(data_path)
        else:
            validate_clean_modality_view(data_path, data_contract)
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
        if model_sha256 is not None:
            actual_model_sha256 = sha256(local_model)
            if actual_model_sha256.lower() != str(model_sha256).lower():
                raise TrainingConfigError(
                    f"模型权重 SHA256 不一致: {actual_model_sha256} != {model_sha256}"
                )
    elif model_sha256 is not None:
        raise TrainingConfigError("model_sha256 只能用于本地 model 权重")
    if initial_weights is not None:
        if model.suffix.lower() not in {".yaml", ".yml"}:
            raise TrainingConfigError("initial_weights 只用于自定义 model YAML")
        if pretrained is not False:
            raise TrainingConfigError("使用 initial_weights 时必须显式设置 pretrained: false")
        initial_path = project_path(initial_weights)
        if not initial_path.is_file():
            raise TrainingConfigError(f"初始化权重不存在: {initial_path}")
        if not initial_weights_sha256:
            raise TrainingConfigError("initial_weights 必须同时提供 initial_weights_sha256")
        actual_sha256 = sha256(initial_path)
        if actual_sha256.lower() != str(initial_weights_sha256).lower():
            raise TrainingConfigError(
                f"初始化权重 SHA256 不一致: {actual_sha256} != {initial_weights_sha256}"
            )
        config["initial_weights"] = str(initial_path)
        if initial_weights_policy not in {"ultralytics_shape_match", P2_INITIALIZATION_POLICY}:
            raise TrainingConfigError(f"未知 initial_weights_policy: {initial_weights_policy}")
    elif initial_weights_sha256 is not None:
        raise TrainingConfigError("initial_weights_sha256 缺少对应的 initial_weights")
    elif initial_weights_policy is not None:
        raise TrainingConfigError("initial_weights_policy 缺少对应的 initial_weights")
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
    mode.add_argument(
        "--smoke-full", action="store_true",
        help="Run one short epoch at the configured image size and batch to catch OOM",
    )
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

    mode_name = (
        "formal" if args.train else "full-resolution-smoke" if args.smoke_full
        else "smoke" if args.smoke else "check-only"
    )
    model_source = str(config.pop("model"))
    initial_weights = config.pop("initial_weights", None)
    initial_weights_policy = config.pop("initial_weights_policy", None)
    print(f"Experiment: {experiment_id}")
    print(f"Mode: {mode_name}")
    print(f"Git commit: {git_commit()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Ultralytics: {ultralytics.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Model: {model_source}")
    if Path(model_source).is_file():
        print(f"Model SHA256: {sha256(Path(model_source))}")
    print(f"Initial weights: {initial_weights or 'embedded in model / none'}")
    if initial_weights is not None:
        print(f"Initial weights policy: {initial_weights_policy}")
    print(f"Data: {config['data']}")

    if not (args.train or args.smoke or args.smoke_full):
        print("配置与本地权重检查通过；未启动训练。")
        return 0

    if args.smoke or args.smoke_full:
        smoke_settings = dict(
            epochs=1,
            patience=1,
            workers=0,
            fraction=0.05,
            plots=False,
            name=experiment_id + ("_SMOKE_FULL" if args.smoke_full else "_SMOKE"),
            exist_ok=True,
        )
        if not args.smoke_full:
            smoke_settings.update(
                batch=min(int(config.get("batch", 1)), 2),
                imgsz=min(int(config.get("imgsz", 640)), 320),
            )
        config.update(smoke_settings)

    started = time.perf_counter()
    model = YOLO(model_source)
    if initial_weights is not None:
        trainer = build_initializing_trainer(Path(initial_weights), initial_weights_policy)
        metrics = model.train(trainer=trainer, **config)
    else:
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
