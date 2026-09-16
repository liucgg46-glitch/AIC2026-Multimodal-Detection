"""Generate a path-safe DEIMv2-S runtime config after verifying source and weights."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train.verify_round5_assets import (  # noqa: E402
    ASSETS, AssetError, DEIMV2_COMMIT, verify_file, verify_git_checkout,
)


def dataset_contract(dataset_root: Path) -> Dict[str, Any]:
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.is_file():
        raise AssetError("缺少 DEIMv2 COCO manifest: %s" % manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("representation") != "rgb_coco_labels_clean_v1"
        or manifest.get("train_count") != 1600
        or manifest.get("val_count") != 400
        or manifest.get("labels_identity", {}).get("aggregate_sha256")
        != "6a670b95b33e803e5d25fc30d7bbd7985cbc4799234c37ff42b3b1c9204025a4"
    ):
        raise AssetError("DEIMv2 COCO manifest 不满足固定 1600/400 labels_clean 合同")
    required = [
        dataset_root / "images/train", dataset_root / "images/val",
        dataset_root / "annotations/instances_train.json",
        dataset_root / "annotations/instances_val.json",
    ]
    if any(not path.exists() for path in required):
        raise AssetError("DEIMv2 COCO 数据视图不完整")
    return manifest


def schedule(profile: str) -> Dict[str, Any]:
    if profile == "smoke":
        return {
            "epoches": 2, "flat_epoch": 1, "no_aug_epoch": 1,
            "policy": [0, 1, 1], "mixup": [0, 1], "stop": 1,
            "copyblend": [0, 1], "matcher": 1,
        }
    return {
        "epoches": 72, "flat_epoch": 34, "no_aug_epoch": 12,
        "policy": [4, 34, 60], "mixup": [4, 34], "stop": 60,
        "copyblend": [4, 60], "matcher": 54,
    }


def build_runtime_config(
    deim_root: Path, dataset_root: Path, backbone: Path, checkpoint: Path, profile: str,
    total_batch_size: int = 8, workers: int = 8,
) -> Dict[str, Any]:
    verify_git_checkout(deim_root, DEIMV2_COMMIT)
    verify_file(backbone, ASSETS["dinov3_vitt_distill"]["sha256"])
    verify_file(checkpoint, ASSETS["deimv2_dinov3_s_coco"]["sha256"])
    dataset_contract(dataset_root)
    upstream = deim_root / "configs/deimv2/deimv2_dinov3_s_coco.yml"
    if not upstream.is_file():
        raise AssetError("固定 DEIMv2 配置缺失: %s" % upstream)
    plan = schedule(profile)
    return {
        "__include__": [upstream.resolve().as_posix()],
        "output_dir": "./outputs/AIC_DEIMV2_DINOV3_S_%s" % profile.upper(),
        "num_classes": 12,
        "remap_mscoco_category": False,
        "DINOv3STAs": {"weights_path": backbone.resolve().as_posix()},
        "epoches": plan["epoches"],
        "flat_epoch": plan["flat_epoch"],
        "no_aug_epoch": plan["no_aug_epoch"],
        "train_dataloader": {
            "total_batch_size": total_batch_size,
            "num_workers": workers,
            "dataset": {
                "img_folder": (dataset_root / "images/train").resolve().as_posix(),
                "ann_file": (dataset_root / "annotations/instances_train.json").resolve().as_posix(),
                "transforms": {"policy": {"epoch": plan["policy"]}},
            },
            "collate_fn": {
                "mixup_epochs": plan["mixup"], "stop_epoch": plan["stop"],
                "copyblend_epochs": plan["copyblend"],
            },
        },
        "val_dataloader": {
            "total_batch_size": total_batch_size,
            "num_workers": workers,
            "dataset": {
                "img_folder": (dataset_root / "images/val").resolve().as_posix(),
                "ann_file": (dataset_root / "annotations/instances_val.json").resolve().as_posix(),
            },
        },
        "DEIMCriterion": {"matcher": {"matcher_change_epoch": plan["matcher"]}},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deim-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path,
                        default=PROJECT_ROOT / "data/processed/deimv2_rgb_coco")
    parser.add_argument("--backbone", type=Path,
                        default=PROJECT_ROOT / ASSETS["dinov3_vitt_distill"]["path"])
    parser.add_argument("--checkpoint", type=Path,
                        default=PROJECT_ROOT / ASSETS["deimv2_dinov3_s_coco"]["path"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "formal"), required=True)
    parser.add_argument("--total-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = build_runtime_config(
            args.deim_root.resolve(), args.dataset_root.resolve(), args.backbone.resolve(),
            args.checkpoint.resolve(),
            args.profile, args.total_batch_size, args.workers,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    except (AssetError, OSError, ValueError, json.JSONDecodeError) as exc:
        print("错误: %s" % exc)
        return 2
    print(json.dumps({
        "config": str(args.output.resolve()), "profile": args.profile,
        "deimv2_commit": DEIMV2_COMMIT,
        "backbone_sha256": ASSETS["dinov3_vitt_distill"]["sha256"],
        "checkpoint_sha256": ASSETS["deimv2_dinov3_s_coco"]["sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
