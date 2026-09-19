"""Check, GPU-smoke or formally train the single-model quality-aware RGB/IR/Depth detector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import ultralytics
import yaml
from ultralytics.cfg import get_cfg

from src.fusion.quality_initialization import M960_CHECKPOINT, M960_SHA256, verify_m960_checkpoint
from src.fusion.quality_trainer import QualityFusionTrainer
from src.fusion.runtime import git_state, require_reviewed_checkout


DEFAULT_CONFIG = ROOT / "configs/experiments/F002_M960_RGB_IR_DEPTH_QUALITY_P345.yaml"
EXPECTED_ARCHITECTURE = {
    "rgb_backbone": "yolo11m_from_RGB_R2_M960_best",
    "auxiliary_pyramids": "small_ir_and_format_aware_depth",
    "fusion_levels": ["P3", "P4", "P5"],
    "masks": ["ir_edge_connected_dark_border", "depth_format_aware_valid_region"],
    "dropout_probability": 0.2,
    "stage": "frozen_rgb_24_layers",
}


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("train"), dict):
        raise ValueError("F002 experiment must contain a train mapping")
    if config.get("architecture") != EXPECTED_ARCHITECTURE:
        raise ValueError("F002 architecture differs from the implemented model")
    if config.get("data_contract") != "canonical_quality11_v1":
        raise ValueError("F002 data contract differs from the implemented dataset")
    if (config.get("initial_checkpoint") != "runs/RGB_R2_M960/weights/best.pt"
            or config.get("initial_checkpoint_sha256") != M960_SHA256):
        raise ValueError("F002 initial M960 checkpoint contract differs")
    train = config["train"]
    required = {
        "model": "yolo11m.yaml",
        "data": "configs/data/F002_RGB_IR_DEPTH_QUALITY11.yaml",
        "seed": 2026,
        "pretrained": False,
        "freeze": 24,
        "imgsz": 960,
        "batch": 4,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "close_mosaic": 0,
        "resume": False,
        "cache": False,
    }
    for key, value in required.items():
        if train.get(key) != value:
            raise ValueError("F002 formal setting differs: " + key)
    if train.get("name") != config.get("experiment_id"):
        raise ValueError("F002 run name differs from experiment_id")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="Run a small real GPU epoch")
    mode.add_argument("--train", action="store_true", help="Run the formal frozen experiment")
    parser.add_argument("--expected-sha", default="")
    args = parser.parse_args()
    config_path = Path(args.config)
    config_path = config_path if config_path.is_absolute() else ROOT / config_path
    config = load_config(config_path)
    state = git_state(ROOT)
    if args.train:
        require_reviewed_checkout(args.expected_sha, state)
    elif args.expected_sha:
        parser.error("--expected-sha only applies to --train")
    verify_m960_checkpoint(M960_CHECKPOINT)
    print(json.dumps({
        "experiment_id": config["experiment_id"],
        "mode": "formal" if args.train else "smoke" if args.smoke else "check-only",
        "git_commit": state[0],
        "git_dirty": state[1],
        "git_branch": state[2],
        "m960_checkpoint_sha256": M960_SHA256,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
    }, sort_keys=True))

    if not (args.smoke or args.train):
        trainer = QualityFusionTrainer.__new__(QualityFusionTrainer)
        trainer.args = get_cfg(overrides=config["train"])
        trainer.data = trainer.get_dataset()
        model = trainer.get_model(verbose=False)
        print("CHECK_ONLY_PASS: pairing, format, M960 RGB transfer, graph, strides=%s" %
              model.stride.tolist())
        return 0

    overrides = dict(config["train"])
    overrides["project"] = str((ROOT / "runs").resolve())
    overrides["data"] = str((ROOT / overrides["data"]).resolve())
    if args.smoke:
        overrides.update(
            epochs=1,
            patience=1,
            batch=2,
            workers=0,
            imgsz=320,
            plots=False,
            name=config["experiment_id"] + "_SMOKE",
            exist_ok=True,
        )
    trainer = QualityFusionTrainer(overrides=overrides)
    trainer.train()
    print("TRAINING_COMPLETE: " + str(trainer.save_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
