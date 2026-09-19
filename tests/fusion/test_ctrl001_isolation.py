import inspect
import random
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import yaml
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import init_seeds

from scripts.train import train_ctrl001
from src.fusion import control
from src.fusion.control import (
    RGBControlDataset,
    RGBControlTrainer,
    canonical_control_records,
    initialize_rgb_control,
)
from src.fusion.dual_stream_model import DualStreamModel
from src.fusion.initialization import initialize_canonical
from src.fusion.paired_dataset import PairedDataset
from src.fusion.runtime import require_reviewed_checkout


ROOT = Path(__file__).resolve().parents[2]
HEAD = "e3ddb3793b4c8e3f8055652157a0f0580fb07521"


def _record(tmp_path):
    yy, xx = np.mgrid[:31, :47]
    rgb_image = np.stack((xx * 3, yy * 5, xx + yy), axis=-1).astype(np.uint8)
    ir_image = np.stack((yy * 2, xx * 4, xx + 2 * yy), axis=-1).astype(np.uint8)
    rgb, ir, label = tmp_path / "rgb.png", tmp_path / "ir.png", tmp_path / "sample.txt"
    assert cv2.imwrite(str(rgb), rgb_image)
    assert cv2.imwrite(str(ir), ir_image)
    label.write_text("0 0.35 0.4 0.2 0.3\n", encoding="utf-8")
    return [("sample", rgb, ir, label)]


def _seed(value=2026):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def test_ctrl001_uses_paired_dataset_and_never_constructs_standard_yolo_dataset(tmp_path, hyp):
    source = inspect.getsource(control)
    assert "YOLODataset" not in source
    assert "YOLO(" not in inspect.getsource(train_ctrl001)
    paired = PairedDataset(_record(tmp_path), 64, hyp, augment=True)
    dataset = RGBControlDataset(paired)
    assert isinstance(dataset.paired_dataset, PairedDataset)
    assert not isinstance(dataset, type(paired))
    assert RGBControlTrainer.train is DetectionTrainer.train


def test_ctrl001_and_f001_transformed_rgb_are_bitwise_equal(tmp_path, hyp):
    hyp.hsv_h, hyp.hsv_s, hyp.hsv_v = 0.015, 0.7, 0.4
    records = _record(tmp_path)
    fused_dataset = PairedDataset(records, 64, hyp, augment=True)
    control_dataset = RGBControlDataset(PairedDataset(records, 64, hyp, augment=True))
    _seed()
    fused = fused_dataset[0]
    _seed()
    rgb_only = control_dataset[0]
    assert torch.equal(rgb_only["img"], fused["img"][:3])
    assert torch.equal(rgb_only["bboxes"], fused["bboxes"])
    assert torch.equal(rgb_only["cls"], fused["cls"])


def test_ctrl001_and_f001_common_model_state_is_bitwise_equal():
    init_seeds(2026, deterministic=True)
    fusion = DualStreamModel(nc=12, verbose=False)
    initialize_canonical(fusion, verbose=False)
    init_seeds(2026, deterministic=True)
    control_model = DetectionModel("yolo11n.yaml", ch=3, nc=12, verbose=False)
    initialize_rgb_control(control_model, verbose=False)
    fusion_state, control_state = fusion.state_dict(), control_model.state_dict()
    common = sorted(key for key in fusion_state if key.startswith("model."))
    assert common == sorted(control_state)
    assert all(torch.equal(fusion_state[key], control_state[key]) for key in common)
    assert all(key.startswith(("ir_encoder.", "fusion4.", "fusion5."))
               for key in set(fusion_state) - set(control_state))


def test_control_records_delegate_to_canonical_hash_split_and_label_validator(monkeypatch):
    records = {
        "train": [("t%04d" % index, None, None, None) for index in range(1600)],
        "val": [("v%04d" % index, None, None, None) for index in range(400)],
    }
    calls = []
    monkeypatch.setattr(control, "canonical_records", lambda: calls.append("validated") or records)
    assert canonical_control_records() is records
    assert calls == ["validated"]
    monkeypatch.setattr(control, "canonical_records", lambda: (_ for _ in ()).throw(ValueError("label hash mismatch")))
    with pytest.raises(ValueError, match="hash"):
        canonical_control_records()


@pytest.mark.parametrize(
    "state,message",
    [
        (("0" * 40, False, "DETACHED"), "does not match"),
        ((HEAD, True, "DETACHED"), "clean working tree"),
        ((HEAD, False, "feature/multimodal-fusion"), "detached HEAD"),
    ],
)
def test_formal_diagnostics_and_ctrl_gate_reject_wrong_dirty_or_attached_checkout(state, message):
    with pytest.raises(RuntimeError, match=message):
        require_reviewed_checkout(HEAD, state)
    for path in (ROOT / "scripts/analysis/analyze_f001_fusion.py",
                 ROOT / "scripts/analysis/audit_f001_alignment.py"):
        assert 'add_argument("--expected-sha", required=True)' in path.read_text(encoding="utf-8")


def test_ctrl001_check_only_never_calls_train(monkeypatch):
    called = []

    class FakeTrainer:
        def get_dataset(self):
            return {"train": "train", "val": "val", "nc": 12, "names": {}}

        def get_model(self, verbose=False):
            del verbose
            return object()

        def train(self):
            called.append("train")

    config = yaml.safe_load((ROOT / "configs/experiments/CTRL001_RGB_F001_AUG.yaml").read_text(encoding="utf-8"))
    monkeypatch.setattr(train_ctrl001, "RGBControlTrainer", FakeTrainer)
    monkeypatch.setattr(train_ctrl001, "load_reviewed_config", lambda path: config)
    monkeypatch.setattr(train_ctrl001, "git_state", lambda root: (HEAD, True, "feature/multimodal-fusion"))
    monkeypatch.setattr(train_ctrl001, "verify_checkpoint", lambda path: "digest")
    monkeypatch.setattr(train_ctrl001, "initialize_rgb_control", lambda model, path, verbose: None)
    assert train_ctrl001.main([]) == 0
    assert called == []


def test_ctrl001_formal_train_rejects_dirty_checkout_before_trainer_creation(monkeypatch):
    config = yaml.safe_load((ROOT / "configs/experiments/CTRL001_RGB_F001_AUG.yaml").read_text(encoding="utf-8"))
    monkeypatch.setattr(train_ctrl001, "load_reviewed_config", lambda path: config)
    monkeypatch.setattr(train_ctrl001, "git_state", lambda root: (HEAD, True, "DETACHED"))
    monkeypatch.setattr(train_ctrl001, "RGBControlTrainer",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("trainer constructed")))
    with pytest.raises(RuntimeError, match="clean working tree"):
        train_ctrl001.main(["--train", "--expected-sha", HEAD])
