"""Read-only F002 epoch-zero protocol check and best.pt modality ablations on fixed val400."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader

from scripts.train.train_quality_fusion import DEFAULT_CONFIG, load_config
from src.fusion.quality_dataset import QualityTriModalDataset
from src.fusion.quality_initialization import file_sha256
from src.fusion.quality_model import QualityTriModalModel
from src.fusion.quality_trainer import QualityFusionTrainer


M960_RECT_VAL_BATCH = 16  # RGB_R2_M960 formal log validates val400 in 25 batches


def val_loader(trainer, rectangle, rgb_protocol_resize=False):
    if not rectangle and not rgb_protocol_resize:
        return trainer.test_loader
    batch = M960_RECT_VAL_BATCH if rectangle else trainer.test_loader.batch_size
    dataset = QualityTriModalDataset(
        trainer.tri_records["val"], trainer.args.imgsz, trainer.args,
        augment=False, rect_batch_size=batch if rectangle else None,
        rgb_protocol_resize=rgb_protocol_resize,
    )
    return build_dataloader(dataset, batch=batch, workers=0, shuffle=False, rank=-1)


def validate_mode(trainer, loader, mode):
    validator = trainer.get_validator()
    validator.dataloader = loader
    original = validator.preprocess

    def preprocess_with_ablation(batch):
        batch = original(batch)
        if mode in {"IR_OFF", "BOTH_OFF"}:
            batch["img"][:, 3:6].zero_()
            batch["img"][:, 9].zero_()
        if mode in {"DEPTH_OFF", "BOTH_OFF"}:
            batch["img"][:, 6:9].zero_()
        return batch

    validator.preprocess = preprocess_with_ablation
    scores = validator(trainer)
    return {
        "map50": float(scores["metrics/mAP50(B)"]),
        "map5095": float(scores["metrics/mAP50-95(B)"]),
        "precision": float(scores["metrics/precision(B)"]),
        "recall": float(scores["metrics/recall(B)"]),
        "mode": mode,
        "images": len(loader.dataset),
    }


def persist_report(path, report):
    staged = path.with_name("." + path.name + ".tmp")
    staged.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    staged.replace(path)


def audit(config_path, best_path, expected_best_sha256, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Audit output already exists: " + str(output))
    config = load_config(config_path)
    overrides = dict(config["train"])
    overrides.update(
        data=str((ROOT / overrides["data"]).resolve()),
        project=str((ROOT / "runs").resolve()),
        name="F002_QUALITY_READONLY_AUDIT",
        exist_ok=True, epochs=1, workers=0, amp=False, plots=False,
    )
    trainer = QualityFusionTrainer(overrides=overrides)
    trainer._setup_train()  # initialize the actual trainer model and full val loaders; no train step
    trainer.loss_items = torch.zeros(3, device=trainer.device)
    trainer.epoch = 0
    loaders = {
        "legacy_square": val_loader(trainer, False),
        "legacy_rectangle": val_loader(trainer, True),
        "rgb_square": val_loader(trainer, False, rgb_protocol_resize=True),
        "rgb_rectangle": val_loader(trainer, True, rgb_protocol_resize=True),
    }
    report = {
        "source_m960_sha256": config["initial_checkpoint_sha256"],
        "best_pt_sha256": None,
        "baseline_m960_original_reval_map5095": 0.47901307551468025,
        "baseline_m960_training_history_best_map5095": 0.47923,
        "square_val_batch": trainer.test_loader.batch_size,
        "rectangle_val_batch": M960_RECT_VAL_BATCH,
        "completed": False,
        "results": {},
    }
    for geometry, loader in loaders.items():
        key = "M960_INIT_" + geometry
        report["results"][key] = validate_mode(trainer, loader, "BOTH_OFF")
        persist_report(output, report)
        print("AUDIT_F002_RESULT=" + json.dumps({key: report["results"][key]}, sort_keys=True))

    if best_path is not None:
        best_path = Path(best_path)
        digest = file_sha256(best_path)
        if digest.lower() != expected_best_sha256.lower():
            raise ValueError("F002 best.pt SHA256 differs: " + digest)
        checkpoint = torch.load(str(best_path), map_location="cpu", weights_only=False)
        model = checkpoint.get("ema") or checkpoint.get("model")
        if not isinstance(model, QualityTriModalModel):
            raise ValueError("F002 best.pt does not contain QualityTriModalModel")
        if isinstance(model.args, dict):
            model.args = get_cfg(overrides=model.args)  # restore .box/.cls/.dfl for training-mode loss
        trainer.ema.ema = model.float().to(trainer.device).eval()
        report["best_pt_sha256"] = digest
        best_modes = {
            "legacy_square": ("NORMAL", "BOTH_OFF"),
            "rgb_square": ("NORMAL",),
            "rgb_rectangle": ("NORMAL", "BOTH_OFF", "IR_OFF", "DEPTH_OFF"),
        }
        for geometry, modes in best_modes.items():
            loader = loaders[geometry]
            for mode in modes:
                key = "F002_BEST_" + geometry + "_" + mode
                report["results"][key] = validate_mode(trainer, loader, mode)
                persist_report(output, report)
                print("AUDIT_F002_RESULT=" + json.dumps({key: report["results"][key]}, sort_keys=True))

    report["completed"] = True
    persist_report(output, report)
    print("AUDIT_F002_COMPLETE=" + str(output))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--best", default="runs/F002_M960_RGB_IR_DEPTH_QUALITY_P345/weights/best.pt")
    parser.add_argument("--best-sha256", default="ede0c7507d9d0cdf4e13323105a6d0d7ee06988a492ea210f7f2fbcd12aeeb62")
    parser.add_argument("--output", default="outputs/analysis/F002_quality_protocol_audit.json")
    args = parser.parse_args()
    best = Path(args.best)
    best = best if best.is_absolute() else ROOT / best
    audit(args.config, best, args.best_sha256, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
