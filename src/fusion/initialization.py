"""Hash-verified canonical initialization, with explicit shape-match accounting."""
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
INITIAL_PATH = ROOT / "weights/yolo11n.pt"
INITIAL_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"


def verify_checkpoint(path=INITIAL_PATH):
    path = Path(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != INITIAL_SHA256:
        raise ValueError("Canonical initial checkpoint SHA256 mismatch: " + actual)
    return actual


def load_matching_state(model, source, verbose=True):
    target = model.state_dict()
    loaded, missing, mismatches, used, selected = [], [], [], set(), {}
    ratios = {k: dict(loaded_keys=0, total_keys=0, loaded_numel=0, total_numel=0)
              for k in ("rgb_encoder", "ir_encoder", "neck_head")}
    for key, tensor in target.items():
        if key.startswith("fusion"):
            missing.append(key)
            continue
        if key.startswith("ir_encoder."):
            src_key = "model." + key[len("ir_encoder."):]
            group = "ir_encoder"
        else:
            src_key = key
            group = "rgb_encoder" if int(key.split(".")[1]) <= 10 else "neck_head"
        count = ratios[group]
        count["total_keys"] += 1
        count["total_numel"] += tensor.numel()
        if src_key not in source:
            missing.append(key)
        elif source[src_key].shape != tensor.shape:
            used.add(src_key)
            mismatches.append(dict(key=key, source_shape=list(source[src_key].shape), target_shape=list(tensor.shape)))
        else:
            used.add(src_key)
            selected[key] = source[src_key]
            loaded.append(key)
            count["loaded_keys"] += 1
            count["loaded_numel"] += tensor.numel()
    unexpected = sorted(set(source) - used)
    for count in ratios.values():
        count["key_ratio"] = count["loaded_keys"] / count["total_keys"]
        count["numel_ratio"] = count["loaded_numel"] / count["total_numel"]
    report = dict(loaded_keys=loaded, missing_keys=missing, unexpected_keys=unexpected,
                  shape_mismatch_keys=mismatches, ratios=ratios,
                  policy="shape-match only; 80->12 class branch mismatches retain seeded initialization; fusion stays zero")
    # Fail closed before mutating if anything except the 80->12 classification branch differs.
    bad_missing = [k for k in missing if not k.startswith("fusion")]
    bad_shapes = [r for r in mismatches if not r["key"].startswith("model.23.cv3.")]
    if bad_missing or bad_shapes or unexpected:
        raise ValueError("Unapproved pretrained partial load: " + json.dumps(report))
    model.load_state_dict(selected, strict=False)  # every omitted key accounted for above
    if verbose:
        print("PRETRAINED_LOAD_REPORT=" + json.dumps(report, sort_keys=True))
    return report


def initialize_canonical(model, path=INITIAL_PATH, verbose=True):
    digest = verify_checkpoint(path)  # authenticate bytes BEFORE pickle deserialization
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    source = checkpoint["model"].float().state_dict()
    report = load_matching_state(model, source, verbose=False)
    # Explicit policy: safe even if called on a previously used fusion model.
    for fusion in (model.fusion4, model.fusion5):
        torch.nn.init.zeros_(fusion.proj.weight)
        torch.nn.init.zeros_(fusion.proj.bias)
        torch.nn.init.zeros_(fusion.gate[0].weight)
        torch.nn.init.zeros_(fusion.gate[0].bias)
    report.update(checkpoint="weights/yolo11n.pt", checkpoint_sha256=digest, initial_gate=0.5)
    model.load_report = report
    if verbose:
        print("PRETRAINED_LOAD_REPORT=" + json.dumps(report, sort_keys=True))
    return report
