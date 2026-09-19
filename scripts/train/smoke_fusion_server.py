"""Post-commit F001 server smoke through real Trainer setup; never starts an epoch."""
import argparse
from copy import deepcopy
from contextlib import nullcontext
import gc
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


def _shutdown_loader(loader=None, iterator=None):
    """Stop worker processes held by PyTorch or Ultralytics loader iterators."""
    candidates = [iterator]
    if loader is not None:
        candidates.extend((getattr(loader, "_iterator", None), getattr(loader, "iterator", None)))
    unique = []
    seen = set()
    for candidate in candidates:
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        unique.append(candidate)

    workers = int(getattr(loader, "num_workers", 0) or 0) if loader is not None else 0
    shutdown_confirmed = bool(getattr(loader, "_f001_workers_shutdown", False))
    for candidate in unique:
        shutdown = getattr(candidate, "_shutdown_workers", None)
        if not callable(shutdown):
            continue
        if not getattr(candidate, "_f001_workers_shutdown", False):
            try:
                shutdown()
            except Exception as error:
                raise RuntimeError("DataLoader worker shutdown failed") from error
            try:
                setattr(candidate, "_f001_workers_shutdown", True)
            except (AttributeError, TypeError):
                # PyTorch 2.4 iterators permit this marker; their shutdown method is also idempotent.
                pass
        shutdown_confirmed = True

    if workers > 0 and not shutdown_confirmed:
        raise RuntimeError("DataLoader has workers but exposes no shutdown-capable iterator")
    if loader is not None:
        try:
            setattr(loader, "_f001_workers_shutdown", True)
        except (AttributeError, TypeError):
            pass
        for attribute in ("_iterator", "iterator"):
            if hasattr(loader, attribute):
                try:
                    setattr(loader, attribute, None)
                except (AttributeError, TypeError):
                    pass
    return True


def _cleanup_runtime_loaders(trainer, train_iter):
    """Synchronize CUDA and close both formal loaders, reporting all cleanup failures."""
    failures = []
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception as error:
            failures.append(("cuda_synchronize", error))
    results = {}
    for name, loader, iterator in (
        ("train", getattr(trainer, "train_loader", None), train_iter),
        ("val", getattr(trainer, "test_loader", None), None),
    ):
        try:
            results[name] = _shutdown_loader(loader, iterator)
        except Exception as error:
            failures.append((name, error))
            results[name] = False
    trainer.train_loader = None
    trainer.test_loader = None
    trainer.validator = None
    if failures:
        names = ", ".join(name for name, _ in failures)
        raise RuntimeError("DataLoader cleanup failed for: %s" % names) from failures[0][1]
    return {
        "dataloader_cleanup_passed": True,
        "train_workers_shutdown": results["train"],
        "val_workers_shutdown": results["val"],
    }


def _formal_validation_first_batch(trainer):
    val_iter = iter(trainer.test_loader)
    try:
        raw_batch = next(val_iter)
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
        batch = validator.preprocess(raw_batch)
        with torch.inference_mode(), inference_context:
            predictions = model(batch["img"])
        result = tuple(batch["img"].shape), type(predictions).__name__
    finally:
        if "model" in locals():
            model.float()
        _shutdown_loader(trainer.test_loader, val_iter)
        del val_iter
    return result


def _optimizer_state_is_finite(optimizer):
    return all(
        torch.isfinite(value).all()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


def _optimizer_step_snapshot(optimizer):
    snapshot = {}
    for parameter, state in optimizer.state.items():
        if "step" not in state:
            continue
        step = state["step"]
        if isinstance(step, torch.Tensor):
            if step.numel() != 1 or not torch.isfinite(step).all():
                raise RuntimeError("AdamW state step marker is invalid")
            step = float(step.detach().cpu())
        else:
            step = float(step)
        snapshot[id(parameter)] = step
    return snapshot


def _step_marker_report(snapshot):
    values = list(snapshot.values())
    return {
        "count": len(values),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "sum": sum(values) if values else 0.0,
    }


def _step_marker_advanced(before, after):
    if any(after.get(key, value) < value for key, value in before.items()):
        raise RuntimeError("AdamW state step marker decreased")
    return any(value > before.get(key, 0.0) for key, value in after.items())


def _optimization_attempt(trainer, train_iter, require_optimizer_state=False):
    state_entries_before = len(trainer.optimizer.state)
    marker_before = _optimizer_step_snapshot(trainer.optimizer)
    if require_optimizer_state and (state_entries_before == 0 or not marker_before):
        raise RuntimeError("Second backward must start with materialized AdamW state")
    if state_entries_before and not _optimizer_state_is_finite(trainer.optimizer):
        raise RuntimeError("AdamW state is non-finite before backward")
    scale_before = float(trainer.scaler.get_scale())
    batch = trainer.preprocess_batch(next(train_iter))
    with autocast(trainer.amp):
        loss, loss_items = trainer.model(batch)
        trainer.loss = loss.sum()
        trainer.loss_items = loss_items
    if not torch.isfinite(trainer.loss).all():
        raise RuntimeError("Non-finite training loss")
    trainer.scaler.scale(trainer.loss).backward()
    trainer.optimizer_step()
    scale_after = float(trainer.scaler.get_scale())
    state_entries_after = len(trainer.optimizer.state)
    marker_after = _optimizer_step_snapshot(trainer.optimizer)
    marker_advanced = _step_marker_advanced(marker_before, marker_after)
    scale_backoff = scale_after < scale_before
    if scale_backoff and marker_advanced:
        raise RuntimeError("GradScaler backed off despite an advanced AdamW step marker")
    amp_skipped = scale_backoff and not marker_advanced
    successful_step = marker_advanced and not scale_backoff
    if not amp_skipped and not successful_step:
        raise RuntimeError("optimizer_step outcome is neither an AMP skip nor a real AdamW step")
    if state_entries_after and not _optimizer_state_is_finite(trainer.optimizer):
        raise RuntimeError("AdamW state is non-finite after optimizer_step")
    result = {
        "loss": float(trainer.loss.detach().cpu()),
        "loss_items_type": type(loss_items).__name__,
        "scaler_scale_before": scale_before,
        "scaler_scale_after": scale_after,
        "optimizer_state_entries_before": state_entries_before,
        "optimizer_state_entries_after": state_entries_after,
        "optimizer_state_step_marker_before": _step_marker_report(marker_before),
        "optimizer_state_step_marker_after": _step_marker_report(marker_after),
        "amp_skipped_optimizer_step": amp_skipped,
        "successful_optimizer_step": successful_step,
    }
    trainer.loss = None
    del batch, loss, loss_items
    torch.cuda.empty_cache()
    return result


def _run_two_successful_optimizer_steps(trainer, train_iter, max_attempts_per_success=8):
    attempts = []
    successful_steps = 0
    skipped_steps = 0
    optimizer_step_calls = 0
    first_success_attempt = None
    second_success_attempt = None
    scaler_initial = float(trainer.scaler.get_scale())
    ema_updates_before = trainer.ema.updates

    for target_success in (1, 2):
        for _ in range(max_attempts_per_success):
            optimizer_step_calls += 1
            result = _optimization_attempt(
                trainer,
                train_iter,
                require_optimizer_state=target_success == 2,
            )
            result["attempt"] = optimizer_step_calls
            attempts.append(result)
            if result["amp_skipped_optimizer_step"]:
                skipped_steps += 1
                continue
            successful_steps += 1
            if target_success == 1:
                first_success_attempt = optimizer_step_calls
                if not trainer.optimizer.state or not _optimizer_state_is_finite(trainer.optimizer):
                    raise RuntimeError("First real AdamW step did not materialize finite optimizer state")
            else:
                second_success_attempt = optimizer_step_calls
            break
        else:
            raise RuntimeError(
                "No successful AdamW step within %d attempts for success %d"
                % (max_attempts_per_success, target_success)
            )

    ema_updates_delta = trainer.ema.updates - ema_updates_before
    if ema_updates_delta != optimizer_step_calls:
        raise RuntimeError("EMA updates must match upstream optimizer_step calls, including AMP skips")
    if successful_steps != 2:
        raise RuntimeError("Smoke requires exactly two successful AdamW steps")
    final_marker = _optimizer_step_snapshot(trainer.optimizer)
    if not final_marker or not _optimizer_state_is_finite(trainer.optimizer):
        raise RuntimeError("Final AdamW state is absent or non-finite")
    return {
        "optimization_attempts": attempts,
        "amp_skipped_steps": skipped_steps,
        "successful_optimizer_steps": successful_steps,
        "optimizer_step_calls": optimizer_step_calls,
        "scaler_initial": scaler_initial,
        "scaler_final": float(trainer.scaler.get_scale()),
        "first_success_attempt": first_success_attempt,
        "second_success_attempt": second_success_attempt,
        "optimizer_state_entries": len(trainer.optimizer.state),
        "optimizer_state_step_marker": _step_marker_report(final_marker),
        "ema_updates_delta": ema_updates_delta,
        "two_training_steps_passed": True,
    }


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
    result = checkpoint, restored_shape, metrics, len(one_loader.dataset), strip_report
    _shutdown_loader(one_loader)
    del validator, one_loader
    return result


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

    final_report = None
    with tempfile.TemporaryDirectory(prefix="f001-runtime-smoke-") as directory:
        original_cwd = Path.cwd()
        trainer = None
        train_iter = None
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
            optimization_report = _run_two_successful_optimizer_steps(trainer, train_iter)

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
                **optimization_report,
                "second_step_loss": optimization_report["optimization_attempts"][-1]["loss"],
                "loss_items_type": optimization_report["optimization_attempts"][-1]["loss_items_type"],
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
        finally:
            try:
                if trainer is not None:
                    cleanup_report = _cleanup_runtime_loaders(trainer, train_iter)
                    if final_report is not None:
                        final_report.update(cleanup_report)
            finally:
                train_iter = None
                trainer = None
                gc.collect()
                torch.cuda.empty_cache()
                os.chdir(str(original_cwd))
    if final_report is None:
        raise RuntimeError("Smoke completed without producing a final report")
    print(json.dumps(final_report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
