"""Check, smoke-test or train checkpoint-initialized P4/P5 fusion."""

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

from src.fusion.pretrained_initialization import (
    DEPTH_CHECKPOINT, DEPTH_SHA256, IR_CHECKPOINT, IR_SHA256, verify_checkpoint,
)
from src.fusion.quality_initialization import M960_CHECKPOINT, M960_SHA256, verify_m960_checkpoint
from src.fusion.pretrained_trainer import PretrainedFusionTrainer
from src.fusion.runtime import git_state, require_reviewed_checkout


DEFAULT_CONFIG = ROOT / "configs/experiments/F003_M960_RGB_IR_PRETRAINED_P45.yaml"


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict) or not isinstance(config.get("train"), dict):
        raise ValueError("Experiment must contain a train mapping")
    if config.get("data_contract") != "canonical_pretrained11_v1":
        raise ValueError("Unexpected data contract")
    if not isinstance(config.get("use_depth"), bool):
        raise ValueError("use_depth must be a boolean")
    required = {
        "model": "yolo11m.yaml", "data": "configs/data/F003_PRETRAINED_RGB_IR_DEPTH.yaml",
        "seed": 2026, "pretrained": False, "freeze": 24, "imgsz": 960,
        "batch": 4, "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0,
        "resume": False, "cache": False,
    }
    for key, value in required.items():
        if config["train"].get(key) != value:
            raise ValueError("Formal setting differs: " + key)
    if config["train"].get("name") != config.get("experiment_id"):
        raise ValueError("Run name differs from experiment_id")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--audit-init", action="store_true",
                      help="Validate the exact zero-residual initialization on val400")
    parser.add_argument("--audit-output", default="")
    parser.add_argument("--expected-sha", default="")
    args = parser.parse_args()
    path = Path(args.config)
    path = path if path.is_absolute() else ROOT / path
    config = load_config(path)
    state = git_state(ROOT)
    if args.train:
        require_reviewed_checkout(args.expected_sha, state)
    elif args.expected_sha:
        parser.error("--expected-sha only applies to --train")
    hashes = {
        "rgb": verify_m960_checkpoint(M960_CHECKPOINT, M960_SHA256),
        "ir": verify_checkpoint(IR_CHECKPOINT, IR_SHA256),
        "depth": (verify_checkpoint(DEPTH_CHECKPOINT, DEPTH_SHA256)
                  if config["use_depth"] else "disabled"),
    }
    print(json.dumps({
        "experiment_id": config["experiment_id"], "use_depth": config["use_depth"],
        "mode": ("formal" if args.train else "smoke" if args.smoke else
                 "audit-init" if args.audit_init else "check-only"),
        "git_commit": state[0], "git_dirty": state[1], "checkpoints": hashes,
        "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
    }, sort_keys=True))

    overrides = dict(config["train"])
    overrides["project"] = str((ROOT / "runs").resolve())
    overrides["data"] = str((ROOT / overrides["data"]).resolve())
    PretrainedFusionTrainer.use_depth = config["use_depth"]
    if not (args.smoke or args.train or args.audit_init):
        trainer = PretrainedFusionTrainer.__new__(PretrainedFusionTrainer)
        trainer.args = get_cfg(overrides=overrides)
        trainer.data = trainer.get_dataset()
        model = trainer.get_model(verbose=False)
        assert model.use_depth == config["use_depth"]
        print("CHECK_ONLY_PASS: pairing, checkpoint transfer and graph construction")
        return 0
    if args.smoke:
        overrides.update(epochs=1, patience=1, batch=1, workers=0, imgsz=320,
                         plots=False, name=config["experiment_id"] + "_SMOKE", exist_ok=True)
    trainer = PretrainedFusionTrainer(overrides=overrides)
    if args.audit_init:
        trainer._setup_train()  # build the real model/val loader without an optimizer step
        trainer.loss_items = torch.zeros(3, device=trainer.device)
        trainer.epoch = 0
        scores = trainer.get_validator()(trainer)
        report = {
            "experiment_id": config["experiment_id"],
            "git_commit": state[0],
            "use_depth": config["use_depth"],
            "images": len(trainer.test_loader.dataset),
            "map50": float(scores["metrics/mAP50(B)"]),
            "map5095": float(scores["metrics/mAP50-95(B)"]),
            "precision": float(scores["metrics/precision(B)"]),
            "recall": float(scores["metrics/recall(B)"]),
        }
        output = (Path(args.audit_output) if args.audit_output else
                  ROOT / "outputs/analysis" / (config["experiment_id"] + "_init_audit.json"))
        output = output if output.is_absolute() else ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("PRETRAINED_FUSION_INIT_AUDIT=" + json.dumps(report, sort_keys=True))
        return 0
    trainer.train()
    print("TRAINING_COMPLETE: " + str(trainer.save_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
