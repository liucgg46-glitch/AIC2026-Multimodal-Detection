"""Read-only F001 ablation and feature-statistics helpers."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.torch_utils import select_device

from scripts.data.prepare_ir_yolo import CLASS_NAMES
from .dual_stream_model import DualStreamModel
from .paired_dataset import PairedDataset, canonical_records


ROOT = Path(__file__).resolve().parents[2]
F001_CHECKPOINT_SHA256 = "f6ee90047ff15f2a2e8b0fc8f4146b55e6e29ec20410b0f5e84d51caff3059cf"
F001_TRAINING_SHA = "e3ddb3793b4c8e3f8055652157a0f0580fb07521"
ABLATION_MODES = ("NORMAL", "RESIDUAL_OFF", "IR_ZERO", "IR_SHUFFLED")
METRIC_KEYS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha(root: Path = ROOT) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def require_new_output_dir(path: Path) -> Path:
    path = path.resolve()
    if path.exists():
        raise FileExistsError("Refusing to overwrite existing analysis output: %s" % path)
    path.mkdir(parents=True)
    return path


@contextmanager
def immutable_checkpoint(path: Path, expected_sha256: str = ""):
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    before = sha256(path)
    if expected_sha256 and before.lower() != expected_sha256.lower():
        raise RuntimeError("F001 checkpoint SHA256 mismatch")
    try:
        yield before
    finally:
        after = sha256(path)
        if after != before:
            raise RuntimeError("F001 checkpoint changed during read-only diagnostics")


def fixed_val_records() -> List[Tuple[str, Path, Path, Path]]:
    records = canonical_records()["val"]
    split = [line.strip() for line in (ROOT / "data/splits/val.txt").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    stems = [record[0] for record in records]
    if len(stems) != 400 or len(set(stems)) != 400 or stems != split:
        raise RuntimeError("Diagnostics require the exact ordered fixed val=400 split")
    return records


def _spatial_shape(path: Path) -> Tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim < 2:
        raise ValueError("Cannot decode image for shape-safe shuffle: %s" % path)
    return int(image.shape[0]), int(image.shape[1])


def deterministic_ir_shuffle(
    records: Sequence[Tuple[str, Path, Path, Path]], seed: int
) -> Tuple[List[Tuple[str, Path, Path, Path]], List[Dict[str, Any]]]:
    """Derange IR paths within exact spatial-shape groups."""
    groups: Dict[Tuple[int, int], List[Tuple[str, Path, Path, Path]]] = {}
    shapes: Dict[str, Tuple[int, int]] = {}
    for record in records:
        stem, rgb_path, ir_path, _ = record
        rgb_shape, ir_shape = _spatial_shape(rgb_path), _spatial_shape(ir_path)
        if rgb_shape != ir_shape:
            raise ValueError("Paired shape mismatch before shuffle: %s" % stem)
        shapes[stem] = rgb_shape
        groups.setdefault(rgb_shape, []).append(record)

    mapping: Dict[str, Tuple[str, Path]] = {}
    rng = random.Random(seed)
    for shape in sorted(groups):
        group = sorted(groups[shape], key=lambda item: item[0])
        if len(group) < 2:
            raise ValueError("Cannot derange singleton shape group: %sx%s" % shape)
        order = list(group)
        rng.shuffle(order)
        shift = rng.randrange(1, len(order))
        for index, target in enumerate(order):
            source = order[(index + shift) % len(order)]
            mapping[target[0]] = (source[0], source[2])
        assigned = [mapping[row[0]][0] for row in group]
        expected = [row[0] for row in group]
        if sorted(assigned) != sorted(expected) or len(set(assigned)) != len(group):
            raise RuntimeError("IR shuffle is not a strict within-shape permutation")

    shuffled = []
    manifest = []
    for stem, rgb_path, ir_path, label_path in records:
        ir_stem, shuffled_ir = mapping[stem]
        if ir_stem == stem or shapes[ir_stem] != shapes[stem]:
            raise RuntimeError("IR shuffle violated derangement or shape-group contract")
        shuffled.append((stem, rgb_path, shuffled_ir, label_path))
        manifest.append({
            "rgb_stem": stem,
            "original_ir_stem": stem,
            "shuffled_ir_stem": ir_stem,
            "height": shapes[stem][0],
            "width": shapes[stem][1],
        })
    return shuffled, manifest


def run_fresh_model_plan(modes, model_loader, mode_runner, statistics_runner):
    """Run every mutating validator boundary against a separately loaded model."""
    reports = []
    for mode in modes:
        model = model_loader()
        reports.append(mode_runner(model, mode))
    statistics_model = model_loader()
    return reports, statistics_runner(statistics_model)


class AblationDataset(Dataset):
    """Apply transport-only inference ablations without touching RGB channels."""
    collate_fn = staticmethod(PairedDataset.collate_fn)

    def __init__(self, dataset: PairedDataset, mode: str):
        if mode not in ABLATION_MODES:
            raise ValueError("Unknown F001 ablation mode: %s" % mode)
        self.dataset = dataset
        self.mode = mode
        self.labels = dataset.labels
        self.records = dataset.records
        self.im_files = dataset.im_files

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.dataset[index]
        if self.mode == "IR_ZERO":
            sample = dict(sample)
            image = sample["img"].clone()
            image[3:6].zero_()
            sample["img"] = image
        return sample


@contextmanager
def ablation_context(model: DualStreamModel, mode: str):
    if mode not in ABLATION_MODES:
        raise ValueError("Unknown F001 ablation mode: %s" % mode)
    handles = []
    if mode == "RESIDUAL_OFF":
        def rgb_only(module: torch.nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
            del module, output
            return inputs[0]

        handles = [model.fusion4.register_forward_hook(rgb_only), model.fusion5.register_forward_hook(rgb_only)]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def load_f001_model(checkpoint: Path, device: str) -> DualStreamModel:
    resolved_device = select_device(device)
    model, _ = load_checkpoint(str(checkpoint), device=resolved_device, fuse=False)
    if not isinstance(model, DualStreamModel):
        raise TypeError("Checkpoint does not contain the complete F001 DualStreamModel")
    return model.float().eval()


def validation_args(config: Dict[str, Any], device: str, batch: int, workers: int) -> Any:
    overrides = dict(config["train"])
    overrides.update({
        "data": str((ROOT / "configs/data/F001_RGB_IR_RAW3.yaml").resolve()),
        "device": device,
        "batch": batch,
        "workers": workers,
        "plots": False,
        "save_json": False,
        "mode": "val",
    })
    return get_cfg(overrides=overrides)


def build_loader(dataset: Dataset, batch: int, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=dataset.collate_fn,
    )


def _metric_payload(validator: DetectionValidator, results: Dict[str, Any]) -> Dict[str, Any]:
    missing = [key for key in METRIC_KEYS.values() if key not in results]
    if missing:
        raise RuntimeError("Validator did not return required metrics: %s" % missing)
    maps = np.asarray(validator.metrics.box.maps, dtype=np.float64).reshape(-1)
    if maps.size != len(CLASS_NAMES):
        raise RuntimeError("Validator did not return 12 per-class AP50-95 values")
    return {
        **{name: float(results[key]) for name, key in METRIC_KEYS.items()},
        "per_class_ap50_95": [
            {"class_id": index, "class_name": CLASS_NAMES[index], "ap50_95": float(value)}
            for index, value in enumerate(maps)
        ],
    }


def run_ablation_mode(
    model: DualStreamModel,
    dataset: AblationDataset,
    mode: str,
    args: Any,
    save_dir: Path,
    checkpoint_sha256: str,
    current_git_sha: str,
    seed: int,
) -> Dict[str, Any]:
    loader = build_loader(dataset, args.batch, args.workers)
    validator = DetectionValidator(loader, save_dir=save_dir, args=args)
    with ablation_context(model, mode):
        results = validator(model=model)
    model.float().eval()
    gt_instances = sum(len(label["cls"]) for label in dataset.labels)
    return {
        "mode": mode,
        "deterministic_seed": seed,
        "sample_count": len(dataset),
        "gt_instance_count": int(gt_instances),
        "checkpoint_sha256": checkpoint_sha256,
        "git_sha": current_git_sha,
        **_metric_payload(validator, results),
    }


def _distribution(values: Iterable[float]) -> Dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {key: None for key in ("mean", "std", "min", "max", "p05", "p25", "p50", "p75", "p95")}
    quantiles = np.percentile(array, (5, 25, 50, 75, 95))
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
    }


def collect_fusion_statistics(
    model: DualStreamModel, dataset: Dataset, args: Any, checkpoint_sha256: str, current_git_sha: str
) -> Dict[str, Any]:
    loader = build_loader(dataset, args.batch, args.workers)
    device = next(model.parameters()).device
    state: Dict[str, Dict[str, Any]] = {
        "P4": {"gates": [], "residual_ratio": [], "delta_ratio": [], "rgb_sq": 0.0,
               "residual_sq": 0.0, "delta_sq": 0.0, "images": []},
        "P5": {"gates": [], "residual_ratio": [], "delta_ratio": [], "rgb_sq": 0.0,
               "residual_sq": 0.0, "delta_sq": 0.0, "images": []},
    }
    seen = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch["img"].to(device, non_blocking=True).float() / 255.0
            rgb_features, ir_features = model.backbone_features(images)
            batch_size = int(images.shape[0])
            stems = [dataset.records[index][0] for index in range(seen, seen + batch_size)]
            for name, index, fusion in (("P4", 6, model.fusion4), ("P5", 10, model.fusion5)):
                rgb, ir = rgb_features[index], ir_features[index]
                gate = fusion.gate(torch.cat((rgb, ir), 1))
                projected = fusion.proj(ir)
                residual = gate * projected
                fused = rgb + residual
                state[name]["gates"].append(gate.detach().float().cpu().reshape(-1).numpy())
                for sample_index, stem in enumerate(stems):
                    rgb_norm = float(torch.linalg.vector_norm(rgb[sample_index]).cpu())
                    residual_norm = float(torch.linalg.vector_norm(residual[sample_index]).cpu())
                    delta_norm = float(torch.linalg.vector_norm(fused[sample_index] - rgb[sample_index]).cpu())
                    denominator = max(rgb_norm, 1e-12)
                    residual_ratio = residual_norm / denominator
                    delta_ratio = delta_norm / denominator
                    state[name]["residual_ratio"].append(residual_ratio)
                    state[name]["delta_ratio"].append(delta_ratio)
                    state[name]["images"].append({
                        "stem": stem,
                        "gate_mean": float(gate[sample_index].mean().cpu()),
                        "gate_std": float(gate[sample_index].std(unbiased=False).cpu()),
                        "gate_min": float(gate[sample_index].min().cpu()),
                        "gate_max": float(gate[sample_index].max().cpu()),
                        "gate_proj_ir_to_rgb_norm": residual_ratio,
                        "fused_minus_rgb_to_rgb_norm": delta_ratio,
                    })
                    state[name]["rgb_sq"] += rgb_norm * rgb_norm
                    state[name]["residual_sq"] += residual_norm * residual_norm
                    state[name]["delta_sq"] += delta_norm * delta_norm
            seen += batch_size
    if seen != len(dataset) or seen != 400:
        raise RuntimeError("Fusion statistics did not traverse the full fixed val=400")

    output: Dict[str, Any] = {
        "sample_count": seen,
        "checkpoint_sha256": checkpoint_sha256,
        "git_sha": current_git_sha,
        "levels": {},
    }
    for name, fusion in (("P4", model.fusion4), ("P5", model.fusion5)):
        values = state[name]
        gates = np.concatenate(values["gates"])
        residual_overall = float(np.sqrt(values["residual_sq"]) / max(np.sqrt(values["rgb_sq"]), 1e-12))
        delta_overall = float(np.sqrt(values["delta_sq"]) / max(np.sqrt(values["rgb_sq"]), 1e-12))
        bias = fusion.proj.bias
        output["levels"][name] = {
            "gate": _distribution(gates),
            "projection_weight_norm": float(torch.linalg.vector_norm(fusion.proj.weight.detach()).cpu()),
            "projection_bias_norm": None if bias is None else float(torch.linalg.vector_norm(bias.detach()).cpu()),
            "gate_proj_ir_to_rgb_norm": {
                "overall": residual_overall,
                "per_image_distribution": _distribution(values["residual_ratio"]),
            },
            "fused_minus_rgb_to_rgb_norm": {
                "overall": delta_overall,
                "per_image_distribution": _distribution(values["delta_ratio"]),
            },
            "per_image": values["images"],
        }
    return output


def compare_ctrl001_to_f001(ctrl_path: Path, f001_path: Path) -> Dict[str, Any]:
    ctrl = yaml.safe_load(ctrl_path.read_text(encoding="utf-8"))
    f001_config = yaml.safe_load(f001_path.read_text(encoding="utf-8"))
    f001 = f001_config["train"]
    control_train = ctrl.get("train", {})
    allowed = {"data", "name"}
    compared = sorted((set(f001) | set(control_train)) - allowed)
    mismatches = {
        key: {"f001": f001.get(key), "ctrl001": control_train.get(key)}
        for key in compared
        if f001.get(key) != control_train.get(key)
    }
    augmentation_keys = (
        "degrees", "translate", "scale", "shear", "perspective", "mosaic", "mixup", "cutmix",
        "copy_paste", "multi_scale", "bgr", "fliplr", "flipud", "hsv_h", "hsv_s", "hsv_v", "close_mosaic",
    )
    augmentation_equal = all(control_train.get(key) == f001.get(key) for key in augmentation_keys)
    data_configuration = ctrl.get("data_configuration", {})
    contract = {
        "no_self_declared_data_contract": "data_contract" not in ctrl,
        "rgb_only_architecture": ctrl.get("architecture", {}).get("model") == "rgb_only_yolo11n"
            and ctrl.get("architecture", {}).get("input_channels") == 3,
        "rgb_yolo11n_graph": control_train.get("model") == "yolo11n.yaml",
        "canonical_pretrained": control_train.get("pretrained") == f001.get("pretrained") == "weights/yolo11n.pt",
        "canonical_pretrained_sha256": ctrl.get("architecture", {}).get("pretrained_sha256")
            == f001_config.get("architecture", {}).get("pretrained_sha256"),
        "shared_paired_transform_code": ctrl.get("architecture", {}).get("paired_transform_then_rgb_drop") is True,
        "dedicated_canonical_data_adapter": control_train.get("data") == "configs/data/CTRL001_RGB_CANONICAL.yaml",
        "canonical_rgb_source": data_configuration.get("rgb") == "data/raw/train/visible",
        "canonical_ir_validation_source": data_configuration.get("ir_validation_source") == "data/raw/train/infrared",
        "labels_clean_source": data_configuration.get("labels") == "data/processed/train/labels_clean",
        "fixed_split_paths": data_configuration.get("train") == "data/splits/train.txt"
            and data_configuration.get("val") == "data/splits/val.txt",
        "fixed_counts": data_configuration.get("counts") == [1600, 400],
    }
    passed = not mismatches and augmentation_equal and all(contract.values())
    return {
        "passed": passed,
        "compared_keys": compared,
        "allowed_train_differences": sorted(allowed),
        "mismatches": mismatches,
        "augmentation_keys": list(augmentation_keys),
        "augmentation_equal": augmentation_equal,
        "control_contract": contract,
    }
