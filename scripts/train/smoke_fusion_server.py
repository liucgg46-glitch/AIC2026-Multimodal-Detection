"""Post-commit F001 server smoke through real Trainer setup; never starts an epoch."""
import argparse
from copy import deepcopy
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
_SMOKE_CONFIG = tempfile.TemporaryDirectory(prefix="f001-ultralytics-smoke-config-")
os.environ["YOLO_CONFIG_DIR"] = _SMOKE_CONFIG.name

import torch
import yaml
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.torch_utils import autocast, strip_optimizer

from src.fusion.dual_stream_model import DualStreamModel
from src.fusion.initialization import INITIAL_PATH
from src.fusion.runtime import (
    git_state,
    one_image_loader,
    resolved_runtime_provenance,
    require_reviewed_checkout,
)
from src.fusion.trainer import FusionTrainer


def _inside(path, directory):
    try:
        Path(path).resolve().relative_to(Path(directory).resolve())
        return True
    except ValueError:
        return False


def _trainer_overrides(config, temporary_root, batch, imgsz):
    overrides = dict(config["train"])
    overrides.update({
        "batch": batch,
        "imgsz": imgsz,
        "device": "0",
        "data": str((ROOT / overrides["data"]).resolve()),
        "pretrained": str(INITIAL_PATH.resolve()),
        "project": str((Path(temporary_root) / "project").resolve()),
        "name": "trainer-batch-%d" % batch,
        "exist_ok": True,
        "save": True,
    })
    return overrides


def _formal_validation_first_batch(trainer):
    raw_batch = next(iter(trainer.test_loader))
    if raw_batch["img"].shape[0] != trainer.test_loader.batch_size:
        raise RuntimeError("Formal validation first batch did not reach the resolved batch size")
    validator = trainer.validator
    validator.device = trainer.device
    use_amp = trainer.device.type != "cpu" and trainer.amp
    model = trainer.ema.ema if trainer.ema is not None else trainer.model
    model.eval()
    if hasattr(validator.args, "quantize"):
        # 8.4.144 keeps validation weights FP32 and enters autocast around inference.
        validator.args.quantize = 16 if use_amp else None
        model.float()
        inference_context = autocast(use_amp, device=trainer.device.type)
    else:
        # 8.3.253 converts both the EMA model and input to FP16 during training validation.
        validator.args.half = use_amp
        model.half() if use_amp else model.float()
        inference_context = nullcontext()
    try:
        batch = validator.preprocess(raw_batch)
        with torch.inference_mode(), inference_context:
            predictions = model(batch["img"])
    finally:
        model.float()
    return tuple(batch["img"].shape), type(predictions).__name__


def _optimizer_state_is_finite(optimizer):
    return all(
        torch.isfinite(value).all()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


def _optimization_cycle(trainer, train_iter, require_optimizer_state=False):
    state_entries_before = len(trainer.optimizer.state)
    if require_optimizer_state and state_entries_before == 0:
        raise RuntimeError("Second backward must start with materialized AdamW state")
    batch = trainer.preprocess_batch(next(train_iter))
    with autocast(trainer.amp):
        loss, loss_items = trainer.model(batch)
        trainer.loss = loss.sum()
        trainer.loss_items = loss_items
    if not torch.isfinite(trainer.loss).all():
        raise RuntimeError("Non-finite training loss")
    trainer.scaler.scale(trainer.loss).backward()
    trainer.optimizer_step()
    state_entries_after = len(trainer.optimizer.state)
    if state_entries_after == 0 or not _optimizer_state_is_finite(trainer.optimizer):
        raise RuntimeError("AdamW state is absent or non-finite after optimizer_step")
    result = {
        "loss": float(trainer.loss.detach().cpu()),
        "loss_items_type": type(loss_items).__name__,
        "optimizer_state_entries_before": state_entries_before,
        "optimizer_state_entries_after": state_entries_after,
    }
    trainer.loss = None
    del batch, loss, loss_items
    torch.cuda.empty_cache()
    return result


def _save_and_reload_checkpoint(trainer, temporary_root, imgsz):
    trainer.ema.update_attr(
        trainer.model,
        include=["yaml", "nc", "args", "names", "stride", "class_weights"],
    )
    trainer.epoch = 0
    trainer.best_fitness = trainer.fitness = 0.0
    trainer.save_model()
    checkpoint = trainer.best
    if not checkpoint.is_file() or not _inside(checkpoint, temporary_root):
        raise RuntimeError("Trainer checkpoint escaped TemporaryDirectory or was not created")
    stripped = strip_optimizer(str(checkpoint))
    if (not stripped or stripped.get("epoch") != -1 or stripped.get("optimizer") is not None
            or stripped.get("ema") is not None or stripped.get("scaler") is not None):
        raise RuntimeError("strip_optimizer did not create a final_eval-compatible checkpoint")

    restored, checkpoint_dict = load_checkpoint(str(checkpoint), device=trainer.device, fuse=False)
    if not isinstance(restored, DualStreamModel):
        raise RuntimeError("load_checkpoint did not restore DualStreamModel")
    with torch.inference_mode():
        restored_output = restored(torch.zeros(1, 6, imgsz, imgsz, device=trainer.device))
    restored_shape = tuple(restored_output[0].shape)
    strip_report = {
        "epoch": checkpoint_dict.get("epoch"),
        "optimizer_removed": checkpoint_dict.get("optimizer") is None,
        "ema_promoted_and_removed": checkpoint_dict.get("ema") is None,
        "scaler_removed": checkpoint_dict.get("scaler") is None,
    }
    del restored, restored_output, checkpoint_dict, stripped
    torch.cuda.empty_cache()

    one_loader = one_image_loader(trainer.test_loader.dataset)
    if len(one_loader.dataset) != 1:
        raise RuntimeError("Metric validator smoke dataset is not exactly one image")
    val_args = deepcopy(trainer.args)
    val_args.batch = 1
    val_args.workers = 0
    val_args.plots = False
    val_args.save_json = False
    val_args.model = str(checkpoint)
    use_amp = trainer.device.type != "cpu" and trainer.amp
    if hasattr(val_args, "quantize"):
        val_args.quantize = 16 if use_amp else None
    else:
        val_args.half = use_amp
    validator = DetectionValidator(
        one_loader,
        save_dir=Path(temporary_root) / "one-image-validator",
        args=val_args,
    )
    metrics = validator(model=str(checkpoint))
    return checkpoint, restored_shape, metrics, len(one_loader.dataset), strip_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--batch", required=True, type=int, choices=(1, 2, 8, 16, 32))
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    state = git_state(ROOT)
    head = require_reviewed_checkout(args.expected_sha, state)
    config = yaml.safe_load(
        (ROOT / "configs/experiments/F001_RGB_IR_GATED_P45_YOLO11N.yaml").read_text(encoding="utf-8")
    )
    requested_optimizer = config["train"]["optimizer"]

    with tempfile.TemporaryDirectory(prefix="f001-runtime-smoke-") as directory:
        original_cwd = Path.cwd()
        try:
            # Upstream AMP checks may provision their own test asset; keep every such artifact temporary.
            os.chdir(directory)
            trainer = FusionTrainer(overrides=_trainer_overrides(config, directory, args.batch, args.imgsz))
            trainer._setup_train()
            if not _inside(trainer.save_dir, directory):
                raise RuntimeError("Trainer save_dir is outside TemporaryDirectory")

            setup_report = resolved_runtime_provenance(trainer, requested_optimizer)
            if setup_report["formal_iterations"] != 2500:
                raise RuntimeError("Formal optimizer iterations must be 2500")
            if setup_report["resolved_optimizer"] != "AdamW":
                raise RuntimeError("optimizer=auto did not resolve to AdamW")
            if setup_report["physical_train_batch"] != args.batch:
                raise RuntimeError("Resolved train batch differs from requested physical batch")
            if setup_report["resolved_val_batch"] != args.batch * 2:
                raise RuntimeError("Resolved detection validation batch is not physical batch * 2")

            trainer.model.train()
            trainer.optimizer.zero_grad()
            train_iter = iter(trainer.train_loader)
            ema_updates_before = trainer.ema.updates
            step1 = _optimization_cycle(trainer, train_iter)
            if trainer.ema.updates != ema_updates_before + 1:
                raise RuntimeError("ModelEMA did not update after optimization step 1")
            step2 = _optimization_cycle(trainer, train_iter, require_optimizer_state=True)
            if trainer.ema.updates != ema_updates_before + 2:
                raise RuntimeError("ModelEMA update delta is not 2")
            if step2["optimizer_state_entries_before"] != step1["optimizer_state_entries_after"]:
                raise RuntimeError("AdamW state was not resident when the second backward started")
            if step2["optimizer_state_entries_after"] < step1["optimizer_state_entries_after"]:
                raise RuntimeError("AdamW state entries disappeared after optimization step 2")

            formal_val_shape, formal_val_output_type = _formal_validation_first_batch(trainer)
            checkpoint, restored_shape, metrics, metric_dataset_length, strip_report = _save_and_reload_checkpoint(
                trainer, directory, args.imgsz
            )
            final_report = resolved_runtime_provenance(trainer, requested_optimizer)
            final_report.update({
                "status": "PASS",
                "sha": head,
                "branch": state[2],
                "batch": args.batch,
                "imgsz": args.imgsz,
                "first_step_loss": step1["loss"],
                "second_step_loss": step2["loss"],
                "loss_items_type": step2["loss_items_type"],
                "optimizer_state_entries_after_step1": step1["optimizer_state_entries_after"],
                "optimizer_state_entries_before_step2": step2["optimizer_state_entries_before"],
                "optimizer_state_entries_after_step2": step2["optimizer_state_entries_after"],
                "ema_updates_delta": trainer.ema.updates - ema_updates_before,
                "two_training_steps_passed": True,
                "formal_val_first_batch_shape": formal_val_shape,
                "formal_val_output_type": formal_val_output_type,
                "one_image_validator_dataset_length": metric_dataset_length,
                "one_image_validator_keys": sorted(metrics),
                "checkpoint_loader": "ultralytics.nn.tasks.load_checkpoint",
                "checkpoint_path_is_temporary": _inside(checkpoint, directory),
                "checkpoint_restored_type": "DualStreamModel",
                "checkpoint_forward_shape": restored_shape,
                "strip_optimizer": strip_report,
                "pretrained_ratios": trainer.model.load_report["ratios"],
                "epoch_started": False,
            })
            print(json.dumps(final_report, indent=2))
        finally:
            os.chdir(str(original_cwd))
    return 0


if __name__ == "__main__":
    sys.exit(main())
