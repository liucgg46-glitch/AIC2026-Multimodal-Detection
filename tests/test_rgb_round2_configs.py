from pathlib import Path

import torch
import yaml
from ultralytics.nn.tasks import DetectionModel


ROOT = Path(__file__).resolve().parents[1]


def read_config(name):
    return yaml.safe_load((ROOT / "configs/experiments" / name).read_text(encoding="utf-8"))


def test_round2_configs_keep_the_s960_contract_and_disable_late_mosaic_close():
    baseline = read_config("AUDIT_S960.yaml")
    for name in ("RGB_R2_S960_CONTROL.yaml", "RGB_R2_M960.yaml", "RGB_R2_S960_P2.yaml"):
        candidate = read_config(name)
        for key in (
            "data", "data_contract", "epochs", "imgsz", "nbs", "optimizer", "lr0", "lrf",
            "weight_decay", "cos_lr", "seed", "deterministic", "translate", "scale", "mosaic",
            "hsv_h", "hsv_s", "hsv_v", "fliplr", "conf", "iou", "max_det",
        ):
            assert candidate[key] == baseline[key], (name, key)
        assert candidate["close_mosaic"] == 0
        assert candidate["batch"] == 8


def test_p2_model_has_four_detection_scales_and_stride_four_output():
    model = DetectionModel(str(ROOT / "configs/models/yolo11s-p2.yaml"), ch=3, nc=12, verbose=False)
    assert model.stride.tolist() == [4.0, 8.0, 16.0, 32.0]
    model.eval()
    with torch.inference_mode():
        predictions, _ = model(torch.zeros(1, 3, 64, 64))
    assert predictions.shape[1] == 16
    assert predictions.shape[2] == 340


def test_p2_semantic_initialization_maps_old_p3_p4_p5_and_leaves_p2_random():
    source = DetectionModel("yolo11s.yaml", ch=3, nc=12, verbose=False)
    target = DetectionModel(str(ROOT / "configs/models/yolo11s-p2.yaml"), ch=3, nc=12, verbose=False)
    before = {key: value.clone() for key, value in target.state_dict().items()}
    selected, report = __import__(
        "scripts.train.train_rgb", fromlist=["build_p2_initial_state"]
    ).build_p2_initial_state(target.state_dict(), source.state_dict())
    target.load_state_dict(selected, strict=False)
    after = target.state_dict()

    assert report["loaded_tensors"] > 400
    assert report["loaded_numel"] / report["target_numel"] > 0.95
    assert torch.equal(after["model.23.conv.weight"], source.state_dict()["model.17.conv.weight"])
    assert torch.equal(after["model.25.cv1.conv.weight"], source.state_dict()["model.19.cv1.conv.weight"])
    assert torch.equal(after["model.29.cv2.1.0.conv.weight"], source.state_dict()["model.23.cv2.0.0.conv.weight"])
    assert torch.equal(after["model.29.dfl.conv.weight"], source.state_dict()["model.23.dfl.conv.weight"])
    assert torch.equal(after["model.19.cv1.conv.weight"], before["model.19.cv1.conv.weight"])
    assert torch.equal(after["model.29.cv3.0.2.weight"], before["model.29.cv3.0.2.weight"])


def test_fixed_p2_retry_keeps_the_formal_config_except_run_identity():
    original = read_config("RGB_R2_S960_P2.yaml")
    retry = read_config("RGB_R3_S960_P2_FIXED_INIT.yaml")
    for key, value in original.items():
        if key not in {"experiment_id", "name"}:
            assert retry[key] == value, key
    assert retry["experiment_id"] == retry["name"] == "RGB_R3_S960_P2_FIXED_INIT"
