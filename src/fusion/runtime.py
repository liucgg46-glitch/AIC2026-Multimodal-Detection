"""Runtime-fidelity helpers shared by the formal entry point and server smoke."""
import json
import math
from pathlib import Path
import re
import subprocess

from torch.utils.data import DataLoader, Subset


SHA40 = re.compile(r"[0-9a-fA-F]{40}")


def git_state(root):
    root = Path(root)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=str(root), text=True).strip())
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=str(root), text=True).strip()
    return commit, dirty, branch or "DETACHED"


def require_reviewed_checkout(expected_sha, state):
    """Require the exact, clean, detached commit used by formal training/server smoke."""
    if not expected_sha or SHA40.fullmatch(expected_sha) is None:
        raise RuntimeError("--expected-sha must be an exact 40-character hexadecimal Git SHA")
    commit, dirty, branch = state
    if commit.lower() != expected_sha.lower():
        raise RuntimeError("HEAD does not match --expected-sha: %s != %s" % (commit, expected_sha))
    if dirty:
        raise RuntimeError("Reviewed execution requires a clean working tree")
    if branch != "DETACHED":
        raise RuntimeError("Reviewed execution requires detached HEAD; current branch is " + branch)
    return commit


def formal_iterations(train_count, batch, nbs, epochs):
    return math.ceil(train_count / max(batch, nbs)) * epochs


def one_image_loader(dataset):
    """Build the metric-smoke loader whose underlying dataset is provably length one."""
    if len(dataset) < 1:
        raise ValueError("One-image validator requires a non-empty validation dataset")
    subset = Subset(dataset, [0])
    return DataLoader(subset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn)


def resolved_runtime_provenance(trainer, requested_optimizer):
    groups = []
    for index, group in enumerate(trainer.optimizer.param_groups):
        parameters = list(group["params"])
        groups.append({
            "index": index,
            "role": group.get("param_group"),
            "tensor_count": len(parameters),
            "parameter_count": sum(parameter.numel() for parameter in parameters),
            "lr": group.get("lr"),
            "weight_decay": group.get("weight_decay"),
        })
    train_batch = trainer.train_loader.batch_size
    val_batch = trainer.test_loader.batch_size
    report = {
        "requested_optimizer": requested_optimizer,
        "resolved_optimizer": type(trainer.optimizer).__name__,
        "optimizer_parameter_groups": groups,
        "optimizer_state_entries": len(trainer.optimizer.state),
        "physical_train_batch": train_batch,
        "resolved_val_batch": val_batch,
        "nbs": trainer.args.nbs,
        "accumulate": trainer.accumulate,
        "nominal_effective_batch": train_batch * trainer.accumulate,
        "formal_iterations": formal_iterations(
            len(trainer.train_loader.dataset), trainer.batch_size, trainer.args.nbs, trainer.epochs
        ),
        "amp_resolved": bool(trainer.amp),
        "scaler": type(trainer.scaler).__name__,
        "ema_enabled": trainer.ema is not None and getattr(trainer.ema, "enabled", True),
        "ema_updates": trainer.ema.updates if trainer.ema is not None else None,
        "scheduler": type(trainer.scheduler).__name__ if trainer.scheduler is not None else None,
    }
    return report


def print_resolved_runtime(trainer, requested_optimizer):
    report = resolved_runtime_provenance(trainer, requested_optimizer)
    print("RESOLVED_RUNTIME_PROVENANCE=" + json.dumps(report, sort_keys=True))
    return report
