import importlib.util
import json
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


def _make_clean_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "CANONICAL_TRAIN_COUNT", 1)
    monkeypatch.setattr(MODULE, "CANONICAL_VAL_COUNT", 1)
    (tmp_path / "data/splits").mkdir(parents=True)
    (tmp_path / "data/splits/train.txt").write_text("train_sample\n", encoding="utf-8")
    (tmp_path / "data/splits/val.txt").write_text("val_sample\n", encoding="utf-8")
    canonical = tmp_path / MODULE.CANONICAL_LABELS
    staged = tmp_path / MODULE.RGB_CLEAN_DATA.parent
    canonical.mkdir(parents=True)
    (staged / "labels/train").mkdir(parents=True)
    (staged / "labels/val").mkdir(parents=True)
    for stem, subset in (("train_sample", "train"), ("val_sample", "val")):
        content = "0 0.5 0.5 0.1 0.1\n"
        (canonical / (stem + ".txt")).write_text(content, encoding="utf-8")
        (staged / "labels" / subset / (stem + ".txt")).write_text(content, encoding="utf-8")
    identity = MODULE.aggregate_labels(list(canonical.glob("*.txt")))
    monkeypatch.setattr(MODULE, "CANONICAL_LABELS_CLEAN_SHA256", identity["aggregate_sha256"])
    (staged / "data.yaml").write_text("names: [person]\n", encoding="utf-8")
    manifest = {
        "representation": "rgb_byte_preserving_clean_labels",
        "train_count": 1,
        "val_count": 1,
        "labels_identity": identity,
    }
    (staged / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return staged


def test_clean_contract_accepts_exact_staged_labels(tmp_path, monkeypatch):
    staged = _make_clean_contract(tmp_path, monkeypatch)
    MODULE.validate_clean_rgb_view(staged / "data.yaml")


def test_clean_contract_rejects_staged_label_tampering(tmp_path, monkeypatch):
    staged = _make_clean_contract(tmp_path, monkeypatch)
    (staged / "labels/train/train_sample.txt").write_text(
        "1 0.5 0.5 0.1 0.1\n", encoding="utf-8"
    )
    with pytest.raises(MODULE.TrainingConfigError, match="不一致"):
        MODULE.validate_clean_rgb_view(staged / "data.yaml")


def test_clean_contract_rejects_broken_manifest(tmp_path, monkeypatch):
    staged = _make_clean_contract(tmp_path, monkeypatch)
    (staged / "manifest.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(MODULE.TrainingConfigError, match="格式损坏"):
        MODULE.validate_clean_rgb_view(staged / "data.yaml")
