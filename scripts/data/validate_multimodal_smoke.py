"""Run side-effect-controlled Ultralytics smoke tests for six modality views."""

import ast
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import subprocess
import tempfile
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
JSON_OUTPUT = PROJECT_ROOT / "outputs" / "analysis" / "modality_smoke_validation.json"
SEED = 2026
IMAGE_SIZE = 640
BATCH_SIZE = 2
WORKERS = 0
DEVICE = "cpu"
MODEL_ARCHITECTURE = "yolo11n.yaml"
MODEL_CLASS_COUNT = 12
MODEL_INPUT_CHANNELS = 3
ALLOWED_GIT_PATHS = {
    "scripts/data/validate_multimodal_smoke.py",
    "tests/test_validate_multimodal_smoke.py",
    "outputs/analysis/modality_smoke_validation.json",
}


class SmokeValidationError(RuntimeError):
    """Raised when a C6 integration invariant is not satisfied."""


class ViewSpec(NamedTuple):
    view: str
    modality: str
    data_yaml: Path


def canonical_views(project_root: Path = PROJECT_ROOT) -> List[ViewSpec]:
    processed = project_root / "data" / "processed"
    depth = processed / "depth_trainable"
    return [
        ViewSpec("ir_raw3", "infrared", processed / "ir_trainable/raw3/data.yaml"),
        ViewSpec("depth_compat8", "depth", depth / "compat8/data.yaml"),
        ViewSpec("depth_validmask", "depth", depth / "validmask/data.yaml"),
        ViewSpec("depth_percentile", "depth", depth / "percentile/data.yaml"),
        ViewSpec("depth_log", "depth", depth / "log/data.yaml"),
        ViewSpec("depth_inverse", "depth", depth / "inverse/data.yaml"),
    ]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeValidationError(message)


def project_relative(path: Path, project_root: Path = PROJECT_ROOT) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise SmokeValidationError("项目文件必须位于 repository root") from exc


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_head_sha(project_root: Path = PROJECT_ROOT) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(project_root), check=True, capture_output=True, text=True
    )
    sha = result.stdout.strip()
    require(len(sha) == 40 and all(char in "0123456789abcdef" for char in sha), "Git HEAD 不是完整 SHA")
    return sha


def git_status_paths(project_root: Path = PROJECT_ROOT) -> List[str]:
    result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=str(project_root),
        check=True,
        capture_output=True,
        text=True,
    )
    paths: List[str] = []
    for line in result.stdout.splitlines():
        if len(line) >= 4:
            paths.append(line[3:].replace("\\", "/"))
    return sorted(paths)


def validate_yaml_contract(path: Path) -> Dict[str, Any]:
    require(path.is_file(), "data.yaml 不存在")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SmokeValidationError("data.yaml 无法解析") from exc
    require(isinstance(data, dict), "data.yaml 必须是 mapping")
    for key in ("train", "val", "names"):
        require(data.get(key) is not None, "data.yaml 缺少 {}".format(key))
    channels = data.get("channels", 3)
    require(channels == MODEL_INPUT_CHANNELS, "data.yaml channels 必须为 3")
    names = data["names"]
    require(isinstance(names, (dict, list)) and len(names) == MODEL_CLASS_COUNT, "data.yaml 类别数必须为 12")

    root_value = data.get("path")
    windows_absolute = isinstance(root_value, str) and PureWindowsPath(root_value).is_absolute()
    linux_portable = not windows_absolute
    warning = None
    if windows_absolute:
        warning = "Windows absolute dataset path; local runtime only, Linux server path requires separate cleanup"
    return {
        "yaml_parse_pass": True,
        "declared_channels": int(channels),
        "declared_class_count": len(names),
        "path_portability": {
            "windows_local": True,
            "linux_portable": linux_portable,
        },
        "portability_warning": warning,
    }


def obtain_first_batch(
    dataset_builder: Callable[[], Any], loader_builder: Callable[[Any], Any]
) -> Tuple[Any, Mapping[str, Any]]:
    try:
        dataset = dataset_builder()
    except Exception as exc:
        raise SmokeValidationError(
            "dataset build failure: {}: {}".format(type(exc).__name__, exc)
        ) from exc
    require(len(dataset) > 0, "dataset 为空")
    try:
        loader = loader_builder(dataset)
        batch = next(iter(loader))
    except StopIteration as exc:
        raise SmokeValidationError("loader 未返回 batch") from exc
    except Exception as exc:
        raise SmokeValidationError("loader build/iteration failure") from exc
    require(isinstance(batch, Mapping), "loader batch 必须是 mapping")
    return dataset, batch


def _finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor.float()).all().item())


def suppress_dataset_cache_write(prefix: str, path: Path, cache: Dict[str, Any], version: str) -> None:
    """Preserve Ultralytics' in-memory cache contract without writing a cache file."""
    del prefix, path
    cache["version"] = version


def summarize_raw_batch(batch: Mapping[str, Any], dataset_size: int) -> Dict[str, Any]:
    image = batch.get("img")
    require(isinstance(image, torch.Tensor), "batch 缺少 img tensor")
    require(image.ndim == 4, "img tensor 必须为 BCHW")
    require(int(image.shape[0]) == BATCH_SIZE, "实际 batch size 不等于 2")
    require(int(image.shape[1]) == MODEL_INPUT_CHANNELS, "img channels 不等于 3")
    require(image.dtype == torch.uint8, "raw loader img dtype 必须为 torch.uint8")
    require(_finite(image), "img tensor 包含非有限值")

    cls = batch.get("cls")
    require(isinstance(cls, torch.Tensor), "batch 缺少 cls tensor")
    require(cls.ndim == 2 and int(cls.shape[1]) == 1, "cls tensor shape 非法")
    require(_finite(cls), "cls tensor 包含非有限值")
    bboxes = batch.get("bboxes")
    require(isinstance(bboxes, torch.Tensor), "batch 缺少 bboxes tensor")
    require(bboxes.ndim == 2 and int(bboxes.shape[1]) == 4, "bboxes tensor shape 非法")
    require(int(bboxes.shape[0]) == int(cls.shape[0]), "cls/bboxes 行数不一致")
    require(_finite(bboxes), "bboxes tensor 包含非有限值")
    require("batch_idx" in batch and isinstance(batch["batch_idx"], torch.Tensor), "batch 缺少 batch_idx")
    require("im_file" in batch and len(batch["im_file"]) == BATCH_SIZE, "batch 缺少 im_file")
    return {
        "batch_obtained": True,
        "dataset_size": int(dataset_size),
        "actual_batch_size": int(image.shape[0]),
        "image_shape": [int(value) for value in image.shape],
        "image_dtype": str(image.dtype),
        "image_min": float(image.min().item()),
        "image_max": float(image.max().item()),
        "image_finite": True,
        "cls_present": True,
        "cls_shape": [int(value) for value in cls.shape],
        "cls_finite": True,
        "bboxes_present": True,
        "bboxes_shape": [int(value) for value in bboxes.shape],
        "bboxes_finite": True,
        "batch_idx_present": True,
        "im_file_present": True,
    }


def clone_batch(batch: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.clone()
        elif isinstance(value, list):
            result[key] = list(value)
        elif isinstance(value, tuple):
            result[key] = tuple(value)
        else:
            result[key] = value
    return result


def summarize_preprocessed_batch(batch: Mapping[str, Any], expected_shape: Sequence[int]) -> Dict[str, Any]:
    image = batch.get("img")
    require(isinstance(image, torch.Tensor), "preprocess 后缺少 img tensor")
    require([int(value) for value in image.shape] == list(expected_shape), "preprocess 改变了 img shape")
    require(image.dtype == torch.float32, "preprocess img dtype 必须为 torch.float32")
    require(_finite(image), "preprocess img 包含非有限值")
    minimum = float(image.min().item())
    maximum = float(image.max().item())
    require(minimum >= 0.0 and maximum <= 1.0, "preprocess img range 超出 [0,1]")
    return {
        "preprocess_completed": True,
        "image_shape": [int(value) for value in image.shape],
        "image_dtype": str(image.dtype),
        "image_min": minimum,
        "image_max": maximum,
        "image_finite": True,
    }


def collect_tensors(value: Any) -> List[torch.Tensor]:
    tensors: List[torch.Tensor] = []
    if isinstance(value, torch.Tensor):
        tensors.append(value)
    elif isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: str(item)):
            tensors.extend(collect_tensors(value[key]))
    elif isinstance(value, (list, tuple)):
        for item in value:
            tensors.extend(collect_tensors(item))
    return tensors


def summarize_forward(output: Any, input_batch_size: int) -> Dict[str, Any]:
    require(output is not None, "model forward 返回 None")
    tensors = collect_tensors(output)
    require(bool(tensors), "model forward 未返回 tensor")
    all_float_finite = all(_finite(tensor) for tensor in tensors if tensor.is_floating_point())
    require(all_float_finite, "model forward 输出包含 NaN/Inf")
    batch_dimension_match = any(tensor.ndim > 0 and int(tensor.shape[0]) == input_batch_size for tensor in tensors)
    require(batch_dimension_match, "forward 输出无法识别对应输入的 batch dimension")
    summaries = []
    for tensor in tensors[:12]:
        summaries.append(
            {
                "shape": [int(value) for value in tensor.shape],
                "dtype": str(tensor.dtype),
                "finite": _finite(tensor) if tensor.is_floating_point() else True,
            }
        )
    return {
        "forward_completed": True,
        "output_type": type(output).__name__,
        "tensor_count": len(tensors),
        "all_float_tensors_finite": True,
        "batch_dimension_match": True,
        "tensor_summaries": summaries,
    }


def reject_pretrained_source(model_source: str) -> None:
    require(not model_source.lower().endswith(".pt"), "C6 禁止 pretrained .pt")
    require(model_source == MODEL_ARCHITECTURE, "C6 只允许冻结的 architecture-only YAML")


def python38_static_compatibility() -> str:
    ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"), filename=str(SCRIPT_PATH), feature_version=(3, 8))
    return "passed"


def sanitize_exception(exc: BaseException) -> str:
    message = " ".join(str(exc).split())[:240]
    message = message.replace(str(PROJECT_ROOT), "<project>").replace(str(PROJECT_ROOT).replace("\\", "/"), "<project>")
    message = re.sub(r"[A-Za-z]:[\\/][^\s]+", "<absolute-path>", message)
    return message or type(exc).__name__


def validate_no_absolute_identity(value: Any) -> None:
    if isinstance(value, str):
        require(not PureWindowsPath(value).is_absolute(), "JSON 禁止 Windows absolute path")
        require(not value.startswith("/"), "JSON 禁止 POSIX absolute path")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            require("timestamp" not in lowered and "uuid" not in lowered, "JSON 禁止 timestamp/UUID field")
            validate_no_absolute_identity(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            validate_no_absolute_identity(item)


def write_deterministic_json(path: Path, payload: Dict[str, Any]) -> None:
    validate_no_absolute_identity(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def file_snapshot(paths: Iterable[Path]) -> Dict[str, Tuple[int, int]]:
    result: Dict[str, Tuple[int, int]] = {}
    for base in paths:
        require(base.exists(), "受保护路径不存在")
        files = [base] if base.is_file() else sorted(path for path in base.rglob("*") if path.is_file())
        for path in files:
            stat = path.stat()
            result[str(path.resolve())] = (int(stat.st_size), int(stat.st_mtime_ns))
    return result


def batch_stage(
    cfg: Any,
    data: Dict[str, Any],
    mode: str,
    expected_size: int,
    build_yolo_dataset: Callable[..., Any],
    build_dataloader: Callable[..., Any],
) -> Tuple[Mapping[str, Any], Dict[str, Any], Dict[str, Any]]:
    set_seed()

    def dataset_builder() -> Any:
        return build_yolo_dataset(
            cfg,
            img_path=data[mode],
            batch=BATCH_SIZE,
            data=data,
            mode=mode,
            rect=False,
            stride=32,
        )

    def loader_builder(dataset: Any) -> Any:
        return build_dataloader(
            dataset, batch=BATCH_SIZE, workers=WORKERS, shuffle=False, device=DEVICE, pin_memory=False
        )

    dataset, raw_batch = obtain_first_batch(dataset_builder, loader_builder)
    require(len(dataset) == expected_size, "{} dataset size 不一致".format(mode))
    raw_summary = summarize_raw_batch(raw_batch, len(dataset))

    from ultralytics.models.yolo.detect.train import DetectionTrainer

    context = SimpleNamespace(device=torch.device(DEVICE), args=SimpleNamespace(multi_scale=0.0))
    processed = DetectionTrainer.preprocess_batch(context, clone_batch(raw_batch))
    processed_summary = summarize_preprocessed_batch(processed, raw_summary["image_shape"])
    return processed, raw_summary, processed_summary


def validate_one_view(
    spec: ViewSpec,
    model: torch.nn.Module,
    check_det_dataset: Callable[..., Dict[str, Any]],
    build_yolo_dataset: Callable[..., Any],
    build_dataloader: Callable[..., Any],
    get_cfg: Callable[..., Any],
    default_cfg: Any,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "view": spec.view,
        "modality": spec.modality,
        "data_yaml": project_relative(spec.data_yaml),
        "view_pass": False,
    }
    stage = "yaml"
    try:
        result.update(validate_yaml_contract(spec.data_yaml))
        stage = "check_det_dataset"
        data = check_det_dataset(str(spec.data_yaml), autodownload=False)
        result["path_portability"]["windows_local"] = bool(
            Path(data["train"]).exists() and Path(data["val"]).exists()
        )
        require(result["path_portability"]["windows_local"], "data.yaml 本地路径无法解析")

        cfg = get_cfg(default_cfg)
        cfg.imgsz = IMAGE_SIZE
        cfg.cache = False
        cfg.rect = False
        cfg.workers = WORKERS
        cfg.task = "detect"
        cfg.seed = SEED
        cfg.deterministic = True

        stage = "train_loader"
        train_batch, train_raw, train_preprocessed = batch_stage(
            cfg, data, "train", 1600, build_yolo_dataset, build_dataloader
        )
        result["train_dataset_size"] = train_raw["dataset_size"]
        result["train_first_batch"] = train_raw
        result["train_preprocessed_batch"] = train_preprocessed

        stage = "model_forward"
        model.eval()
        with torch.inference_mode():
            output = model(train_batch["img"])
        result["forward_result"] = summarize_forward(output, BATCH_SIZE)

        stage = "val_loader"
        _, val_raw, val_preprocessed = batch_stage(
            cfg, data, "val", 400, build_yolo_dataset, build_dataloader
        )
        result["val_dataset_size"] = val_raw["dataset_size"]
        result["val_first_batch"] = val_raw
        result["val_preprocessed_batch"] = val_preprocessed
        result["view_pass"] = True
    except Exception as exc:
        result["failure_stage"] = stage
        result["exception_type"] = type(exc).__name__
        result["exception_message"] = sanitize_exception(exc)
    return result


def build_base_payload(head_sha: str) -> Dict[str, Any]:
    return {
        "schema": "aic2026.modality_smoke_validation.v1",
        "git_head_sha": head_sha,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "ultralytics_version": importlib.metadata.version("ultralytics"),
        "device": DEVICE,
        "cuda_available": bool(torch.cuda.is_available()),
        "seed": SEED,
        "imgsz": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "workers": WORKERS,
        "model_architecture": MODEL_ARCHITECTURE,
        "model_class_count": MODEL_CLASS_COUNT,
        "model_input_channels": MODEL_INPUT_CHANNELS,
        "weights_source": "none",
        "random_initialization": True,
        "training_started": False,
        "backward_executed": False,
        "optimizer_step_executed": False,
        "checkpoint_written": False,
        "accuracy_evaluated": False,
        "predict_executed": False,
        "label_cache_write_suppressed": True,
        "font_download_suppressed": True,
        "python38_static_compatibility": python38_static_compatibility(),
        "python38_runtime_validated": False,
    }


def run_validation(project_root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    reject_pretrained_source(MODEL_ARCHITECTURE)
    protected = {
        "data_raw_unchanged": project_root / "data/raw",
        "labels_clean_unchanged": project_root / "data/processed/train/labels_clean",
        "train_split_unchanged": project_root / "data/splits/train.txt",
        "val_split_unchanged": project_root / "data/splits/val.txt",
    }
    for spec in canonical_views(project_root):
        protected[spec.view + "_staging_unchanged"] = spec.data_yaml.parent
    before = {name: file_snapshot([path]) for name, path in protected.items()}

    payload = build_base_payload(git_head_sha(project_root))
    views: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="aic2026-c6-") as temporary:
        os.environ["YOLO_CONFIG_DIR"] = temporary
        os.environ["MPLCONFIGDIR"] = str(Path(temporary) / "matplotlib")

        from ultralytics.cfg import DEFAULT_CFG, get_cfg
        from ultralytics.data import dataset as dataset_module
        from ultralytics.data import utils as data_utils
        from ultralytics.data.build import build_dataloader, build_yolo_dataset
        from ultralytics.data.utils import check_det_dataset
        from ultralytics.nn.tasks import DetectionModel

        dataset_module.save_dataset_cache_file = suppress_dataset_cache_write
        data_utils.check_font = lambda *args, **kwargs: None
        set_seed()
        model = DetectionModel(
            MODEL_ARCHITECTURE,
            nc=MODEL_CLASS_COUNT,
            ch=MODEL_INPUT_CHANNELS,
            verbose=False,
        ).to(torch.device(DEVICE))
        for spec in canonical_views(project_root):
            views.append(
                validate_one_view(
                    spec,
                    model,
                    check_det_dataset,
                    build_yolo_dataset,
                    build_dataloader,
                    get_cfg,
                    DEFAULT_CFG,
                )
            )

    after = {name: file_snapshot([path]) for name, path in protected.items()}
    source_safety = {name: before[name] == after[name] for name in protected}
    payload["source_and_staging_safety"] = source_safety
    payload["views"] = views
    payload["local_runtime_pass"] = all(view["view_pass"] for view in views) and all(source_safety.values())
    return payload


def main() -> None:
    payload = run_validation()
    write_deterministic_json(JSON_OUTPUT, payload)
    unexpected = [path for path in git_status_paths() if path not in ALLOWED_GIT_PATHS]
    require(not unexpected, "C6 出现非预期 Git 文件: {}".format(unexpected))
    require(payload["local_runtime_pass"], "至少一个 C6 view 或 source-safety check 失败")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
