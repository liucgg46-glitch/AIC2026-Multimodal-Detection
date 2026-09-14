import importlib.util
from pathlib import Path

import pytest
import yaml
import subprocess

SPEC = importlib.util.spec_from_file_location(
    "train_rgb_config", Path(__file__).resolve().parents[1] / "scripts/train/train_rgb.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_yaml_pretrained_true_does_not_silently_train_from_scratch(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "PROJECT_ROOT", tmp_path)
    (tmp_path / "data.yaml").write_text("names: [person]\n")
    config = dict(experiment_id="audit", model="yolo11n.yaml", data="data.yaml", pretrained=True)
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(MODULE.TrainingConfigError, match="COCO"):
        MODULE.load_config(path)
    config["pretrained"] = False
    path.write_text(yaml.safe_dump(config))
    assert MODULE.load_config(path)["pretrained"] is False
    config["pretrained"] = "weights.pt"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(MODULE.TrainingConfigError, match="不存在"):
        MODULE.load_config(path)
    (tmp_path / "weights.pt").write_bytes(b"fixture")
    assert MODULE.load_config(path)["pretrained"] == str(tmp_path / "weights.pt")


def test_formal_checkout_rejects_invalid_sha():
    with pytest.raises(MODULE.TrainingConfigError, match="40 位"):
        MODULE.require_formal_checkout("abc")


def test_main_defaults_to_check_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(MODULE, "PROJECT_ROOT", tmp_path)
    (tmp_path / "data.yaml").write_text("names: [person]\n")
    (tmp_path / "weights.pt").write_bytes(b"fixture")
    config = dict(experiment_id="check", model="weights.pt", data="data.yaml", pretrained=True)
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(MODULE.sys, "argv", ["train_rgb.py", "--config", str(path)])
    assert MODULE.main() == 0
    assert "未启动训练" in capsys.readouterr().out
