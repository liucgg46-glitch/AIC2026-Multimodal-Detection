import importlib.util
import json
from pathlib import Path

import pytest
import yaml
import subprocess
import torch

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


def test_local_model_sha256_is_mandatory_when_declared(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "PROJECT_ROOT", tmp_path)
    (tmp_path / "data.yaml").write_text("names: [person]\n")
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"fixed model")
    config = dict(
        experiment_id="hashed", model="weights.pt", data="data.yaml", pretrained=True,
        model_sha256=MODULE.sha256(weights),
    )
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))
    loaded = MODULE.load_config(path)
    assert loaded["model"] == str(weights)
    assert "model_sha256" not in loaded
    config["model_sha256"] = "0" * 64
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(MODULE.TrainingConfigError, match="模型权重 SHA256"):
        MODULE.load_config(path)


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


def test_ir_modality_contract_requires_clean_labels_and_exact_view(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "CANONICAL_TRAIN_COUNT", 1)
    monkeypatch.setattr(MODULE, "CANONICAL_VAL_COUNT", 1)
    split = tmp_path / "data/splits"
    split.mkdir(parents=True)
    (split / "train.txt").write_text("train_sample\n", encoding="utf-8")
    (split / "val.txt").write_text("val_sample\n", encoding="utf-8")
    canonical = tmp_path / MODULE.CANONICAL_LABELS
    canonical.mkdir(parents=True)
    staged = tmp_path / "data/processed/ir_trainable/raw3"
    for subset, stem in (("train", "train_sample"), ("val", "val_sample")):
        (staged / "labels" / subset).mkdir(parents=True)
        (staged / "images" / subset).mkdir(parents=True)
        (staged / "images" / subset / (stem + ".png")).write_bytes(b"fixture")
        label = "0 0.5 0.5 0.1 0.1\n"
        (canonical / (stem + ".txt")).write_text(label, encoding="utf-8")
        (staged / "labels" / subset / (stem + ".txt")).write_text(label, encoding="utf-8")
    identity = MODULE.aggregate_labels(list(canonical.glob("*.txt")))
    monkeypatch.setattr(MODULE, "CANONICAL_LABELS_CLEAN_SHA256", identity["aggregate_sha256"])
    (staged / "data.yaml").write_text("names: [person]\n", encoding="utf-8")
    (staged / "manifest.json").write_text(json.dumps({
        "representation": "raw3", "train_count": 1, "val_count": 1,
        "label_dir": MODULE.CANONICAL_LABELS.as_posix(),
    }), encoding="utf-8")
    MODULE.validate_clean_modality_view(staged / "data.yaml", MODULE.IR_RAW3_CLEAN_CONTRACT)
    (staged / "labels/val/val_sample.txt").write_text("1 0.5 0.5 0.1 0.1\n")
    with pytest.raises(MODULE.TrainingConfigError, match="内容不一致"):
        MODULE.validate_clean_modality_view(staged / "data.yaml", MODULE.IR_RAW3_CLEAN_CONTRACT)


def test_depth_log_contract_rejects_unrecorded_jpg_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "PROJECT_ROOT", tmp_path)
    staged = tmp_path / "data/processed/depth_trainable/log"
    staged.mkdir(parents=True)
    (staged / "data.yaml").write_text("names: [person]\n")
    (staged / "manifest.json").write_text(json.dumps({
        "representation": "log", "train_count": 1600, "val_count": 400,
        "label_dir": MODULE.CANONICAL_LABELS.as_posix(),
        "train_jpg_count": 122, "val_jpg_count": 27,
        "jpg_policy": {"physical_unit": "mm"},
    }))
    with pytest.raises(MODULE.TrainingConfigError, match="PNG/JPG"):
        MODULE.validate_clean_modality_view(staged / "data.yaml", MODULE.DEPTH_LOG_CLEAN_CONTRACT)


def test_custom_yaml_requires_hashed_explicit_initial_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "PROJECT_ROOT", tmp_path)
    (tmp_path / "data.yaml").write_text("names: [person]\n", encoding="utf-8")
    (tmp_path / "model.yaml").write_text("nc: 1\n", encoding="utf-8")
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"trusted fixture")
    config = {
        "experiment_id": "p2",
        "model": "model.yaml",
        "data": "data.yaml",
        "pretrained": False,
        "initial_weights": "weights.pt",
        "initial_weights_sha256": MODULE.sha256(weights),
        "initial_weights_policy": "ultralytics_shape_match",
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    loaded = MODULE.load_config(path)
    assert loaded["model"] == str(tmp_path / "model.yaml")
    assert loaded["initial_weights"] == str(weights)
    assert "initial_weights_sha256" not in loaded

    config["initial_weights_sha256"] = "0" * 64
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(MODULE.TrainingConfigError, match="SHA256"):
        MODULE.load_config(path)


def test_initial_weights_rejects_ambiguous_pretrained_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "PROJECT_ROOT", tmp_path)
    (tmp_path / "data.yaml").write_text("names: [person]\n", encoding="utf-8")
    (tmp_path / "model.yaml").write_text("nc: 1\n", encoding="utf-8")
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"fixture")
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump({
        "experiment_id": "p2", "model": "model.yaml", "data": "data.yaml",
        "pretrained": True, "initial_weights": "weights.pt",
        "initial_weights_sha256": MODULE.sha256(weights),
        "initial_weights_policy": "ultralytics_shape_match",
    }), encoding="utf-8")
    with pytest.raises(MODULE.TrainingConfigError, match="pretrained: false"):
        MODULE.load_config(path)


def test_initial_weights_are_applied_to_trainer_created_model(tmp_path, monkeypatch):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fixture")
    fresh = torch.nn.Linear(1, 1, bias=False)
    preview = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        fresh.weight.fill_(0)
        preview.weight.fill_(0)

    def fresh_model(self, cfg=None, weights=None, verbose=True):
        assert weights is None
        return fresh

    def apply_initial_weights(model, path):
        assert model is fresh
        assert path == checkpoint
        with torch.no_grad():
            model.weight.fill_(7)
        return {"loaded_numel_ratio": 0.965}

    monkeypatch.setattr(MODULE.DetectionTrainer, "get_model", fresh_model)
    monkeypatch.setattr(MODULE, "initialize_p2_model", apply_initial_weights)
    trainer_type = MODULE.build_initializing_trainer(checkpoint, MODULE.P2_INITIALIZATION_POLICY)
    trainer = trainer_type.__new__(trainer_type)
    actual = trainer.get_model(cfg="model.yaml", weights=None, verbose=False)

    assert actual is fresh
    assert fresh.weight.item() == 7
    assert preview.weight.item() == 0


def test_initializing_trainer_rejects_competing_trainer_weights(tmp_path, monkeypatch):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fixture")
    trainer_type = MODULE.build_initializing_trainer(checkpoint, MODULE.P2_INITIALIZATION_POLICY)
    trainer = trainer_type.__new__(trainer_type)
    with pytest.raises(MODULE.TrainingConfigError, match="同时"):
        trainer.get_model(weights=object())
