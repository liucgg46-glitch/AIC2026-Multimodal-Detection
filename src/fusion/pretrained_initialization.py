"""Hash-verified loading of RGB, IR and Depth checkpoints for fusion V3."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .quality_initialization import ROOT, M960_CHECKPOINT, M960_SHA256, file_sha256, load_m960_rgb


IR_CHECKPOINT = ROOT / "runs/IR_R3_M960_RAW3_CLEAN/weights/best.pt"
IR_SHA256 = "1bf27d36eefc5590bcb61b615597e9be25aab37c0b55831b68c4d26885fa96f6"
DEPTH_CHECKPOINT = ROOT / "runs/DEPTH_R4_M960_LOG_LR3E4/weights/best.pt"
DEPTH_SHA256 = "3a0f6509f6bd1292c2ae1f2193b4768cbbf206e8f3d1b937aa8d3481db4411f6"


def verify_checkpoint(path, expected):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Required best.pt is missing: " + str(path))
    actual = file_sha256(path)
    if actual.lower() != expected.lower():
        raise ValueError("Checkpoint SHA256 mismatch for %s: %s" % (path, actual))
    return actual


def load_backbone(backbone, path, expected):
    digest = verify_checkpoint(path, expected)
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    source_model = checkpoint.get("ema") or checkpoint.get("model")
    if source_model is None:
        raise ValueError("Checkpoint has no model/ema: " + str(path))
    source = source_model.float().state_dict()
    selected = {
        "blocks." + key[len("model."):]: value
        for key, value in source.items()
        if key.startswith("model.") and int(key.split(".")[1]) <= 10
    }
    target = backbone.state_dict()
    if set(selected) != set(target):
        raise ValueError("Backbone keyset mismatch: missing=%s extra=%s" % (
            sorted(set(target) - set(selected))[:5], sorted(set(selected) - set(target))[:5]
        ))
    bad = [key for key in target if target[key].shape != selected[key].shape]
    if bad:
        raise ValueError("Backbone tensor shape mismatch: " + str(bad[:5]))
    backbone.load_state_dict(selected, strict=True)
    return {"checkpoint": str(path), "sha256": digest, "tensors": len(target)}


def initialize_pretrained_fusion(model):
    rgb = load_m960_rgb(model, M960_CHECKPOINT, M960_SHA256)
    report = {
        "rgb": rgb,
        "ir": load_backbone(model.ir_backbone, IR_CHECKPOINT, IR_SHA256),
        "depth": (load_backbone(model.depth_backbone, DEPTH_CHECKPOINT, DEPTH_SHA256)
                  if model.use_depth else "disabled"),
        "trainable": "zero-start P4/P5 spatial residual fusion only",
    }
    print("PRETRAINED_FUSION_INITIALIZATION=" + json.dumps(report, sort_keys=True))
    return report
