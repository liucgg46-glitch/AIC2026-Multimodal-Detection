"""Strict, hash-verified transfer of the trained M960 RGB detector into fusion V2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
M960_CHECKPOINT = ROOT / "runs/RGB_R2_M960/weights/best.pt"
M960_SHA256 = "097852eb7357dc8b7db72fef82e0485e5542a67b051977451edd4305be70ac85"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_m960_checkpoint(path=M960_CHECKPOINT, expected=M960_SHA256):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("M960 best.pt is missing: " + str(path))
    actual = file_sha256(path)
    if actual.lower() != expected.lower():
        raise ValueError("M960 best.pt SHA256 mismatch: " + actual)
    return actual


def load_m960_rgb(model, path=M960_CHECKPOINT, expected=M960_SHA256):
    """Require every RGB backbone, neck and detector tensor to match exactly."""
    digest = verify_m960_checkpoint(path, expected)
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    source_model = checkpoint.get("ema") or checkpoint.get("model")
    if source_model is None:
        raise ValueError("M960 checkpoint has no model/ema")
    source = source_model.float().state_dict()
    target = model.state_dict()
    rgb_keys = {key for key in target if key.startswith("model.")}
    if set(source) != rgb_keys:
        raise ValueError("M960 RGB keyset differs: missing=%s, extra=%s" % (
            sorted(rgb_keys - set(source))[:5], sorted(set(source) - rgb_keys)[:5]
        ))
    bad_shapes = [key for key in rgb_keys if target[key].shape != source[key].shape]
    if bad_shapes:
        raise ValueError("M960 RGB tensor shape differs: " + str(bad_shapes[:5]))
    model.load_state_dict(source, strict=False)
    report = {
        "checkpoint": str(path),
        "checkpoint_sha256": digest,
        "rgb_tensors": len(rgb_keys),
        "rgb_numel": sum(target[key].numel() for key in rgb_keys),
        "auxiliary_initialization": "small random pyramids, zero residual projections",
        "policy": "strict_m960_rgb_v1",
    }
    print("QUALITY_FUSION_RGB_INITIALIZATION=" + json.dumps(report, sort_keys=True))
    return report
