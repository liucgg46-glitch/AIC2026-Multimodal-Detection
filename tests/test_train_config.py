"""Check E002 reaches the existing entry point without training or downloading."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/E002_IR_YOLO11N_CLEAN.yaml"


def test_e002_uses_initial_weights_and_full_canonical_ir_data():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["model"] == "weights/yolo11n.pt"
    assert config["data"] == "data/processed/ir_trainable/raw3/data.yaml"
    assert config["resume"] is False
    assert config["pretrained"] is True
    assert config["fraction"] == 1.0
    assert config["optimizer"] == "auto"
    assert config["exist_ok"] is False


@pytest.fixture
def entry(monkeypatch, tmp_path):
    model = Mock()
    model.trainer.save_dir = tmp_path / "runs/E002_IR_YOLO11N_CLEAN"
    model.train.return_value = SimpleNamespace(results_dict={})
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        __version__="test", version=SimpleNamespace(cuda=None),
        cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(
        YOLO=factory, __version__="test"))
    spec = importlib.util.spec_from_file_location("train_config_test", ROOT / "scripts/train/train_rgb.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "git_commit", lambda: "test-only-sha")
    monkeypatch.setattr(module, "configure_console_encoding", lambda: None)
    return module, factory, model


def test_existing_entry_passes_all_e002_parameters_without_metadata(entry, monkeypatch, tmp_path):
    module, factory, model = entry
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for key in ("data", "model"):
        path = tmp_path / config[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test placeholder; never loaded")
    monkeypatch.setattr(module, "parse_args", lambda: SimpleNamespace(config=str(CONFIG)))
    assert module.main() == 0
    factory.assert_called_once_with(str(tmp_path / config["model"]))
    expected = {k: v for k, v in config.items() if k not in {"experiment_id", "model"}}
    expected["data"] = str(tmp_path / config["data"])
    expected["project"] = str(tmp_path / config["project"])
    model.train.assert_called_once_with(**expected)


def test_missing_ir_dataset_fails_before_model_creation(entry, monkeypatch):
    module, factory, model = entry
    monkeypatch.setattr(module, "parse_args", lambda: SimpleNamespace(config=str(CONFIG)))
    assert module.main() == 2
    factory.assert_not_called()
    model.train.assert_not_called()


def test_historical_loss_and_validation_contract():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    expected = dict(box=7.5, cls=0.5, dfl=1.5, single_cls=False, compile=False,
                    profile=False, split="val", conf=None, iou=0.7, max_det=300,
                    half=False, dnn=False, augment=False, agnostic_nms=False,
                    classes=None, optimizer="auto", lr0=0.01, lrf=0.01,
                    cos_lr=False, close_mosaic=10)
    for key, value in expected.items():
        assert key in config and config[key] == value, key


@pytest.mark.parametrize("weight_state", ["missing", "empty", "present"])
def test_e002_preflight_never_downloads(entry, monkeypatch, tmp_path, weight_state):
    module, factory, model = entry
    monkeypatch.setitem(sys.modules, "train_rgb", module)
    spec = importlib.util.spec_from_file_location("e002_preflight_test", ROOT / "scripts/train/train_e002.py")
    preflight = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preflight)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data = tmp_path / config["data"]
    data.parent.mkdir(parents=True)
    data.write_bytes(b"placeholder")
    weight = tmp_path / config["model"]
    if weight_state != "missing":
        weight.parent.mkdir(parents=True)
        weight.write_bytes(b"mock artifact" if weight_state == "present" else b"")
    monkeypatch.setattr(module, "parse_args", lambda: SimpleNamespace(config=str(CONFIG)))
    assert preflight.main() == (0 if weight_state == "present" else 2)
    if weight_state == "present":
        factory.assert_called_once_with(str(weight))
    else:
        factory.assert_not_called()
        model.train.assert_not_called()
