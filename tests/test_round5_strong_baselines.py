from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


COCO = load_script("round5_coco", "scripts/data/prepare_deimv2_coco.py")
ASSETS = load_script("round5_assets", "scripts/train/verify_round5_assets.py")
RUNTIME = load_script("round5_runtime", "scripts/train/prepare_deimv2_runtime.py")
IR = load_script("round5_ir", "scripts/analysis/audit_ir_heat_residual.py")


def test_round5_yolo11x_config_is_hashed_clean_rgb_x1280():
    config = yaml.safe_load((ROOT / "configs/experiments/RGB_R5_X1280.yaml").read_text(encoding="utf-8"))
    assert config["model"] == "weights/yolo11x.pt"
    assert config["model_sha256"] == ASSETS.ASSETS["yolo11x"]["sha256"]
    assert config["data"] == "data/processed/rgb_yolo_clean/data.yaml"
    assert config["data_contract"] == "rgb_labels_clean_v1"
    assert config["imgsz"] == 1280 and config["batch"] == 2


def test_asset_verifier_rejects_changed_bytes(tmp_path):
    path = tmp_path / "asset.pt"
    path.write_bytes(b"trusted")
    expected = ASSETS.sha256(path)
    assert ASSETS.verify_file(path, expected) == expected
    path.write_bytes(b"changed")
    with pytest.raises(ASSETS.AssetError, match="SHA256"):
        ASSETS.verify_file(path, expected)


def test_coco_export_preserves_split_and_uses_zero_based_categories(tmp_path, monkeypatch):
    visible, labels = tmp_path / "visible", tmp_path / "labels"
    visible.mkdir(); labels.mkdir()
    for stem, class_id in (("train", 0), ("val", 11)):
        assert cv2.imwrite(str(visible / (stem + ".png")), np.full((20, 40, 3), 127, np.uint8))
        (labels / (stem + ".txt")).write_text("%d 0.5 0.5 0.5 0.5\n" % class_id)
    train_split, val_split = tmp_path / "train.txt", tmp_path / "val.txt"
    train_split.write_text("train\n"); val_split.write_text("val\n")
    monkeypatch.setattr(COCO, "TRAIN_COUNT", 1)
    monkeypatch.setattr(COCO, "VAL_COUNT", 1)
    monkeypatch.setattr(COCO, "EXPECTED_LABEL_SHA256", COCO.aggregate_labels(labels.glob("*.txt"))["aggregate_sha256"])
    output = tmp_path / "coco"
    manifest = COCO.build_coco_view(visible, labels, train_split, val_split, output, link_mode="copy")
    train = json.loads((output / "annotations/instances_train.json").read_text())
    val = json.loads((output / "annotations/instances_val.json").read_text())
    assert train["annotations"][0]["category_id"] == 0
    assert val["annotations"][0]["category_id"] == 11
    assert train["annotations"][0]["bbox"] == [10.0, 5.0, 20.0, 10.0]
    assert manifest["train_count"] == manifest["val_count"] == 1


def test_runtime_config_overrides_custom_dataset_and_schedule(tmp_path, monkeypatch):
    deim = tmp_path / "DEIMv2"
    upstream = deim / "configs/deimv2/deimv2_dinov3_s_coco.yml"
    upstream.parent.mkdir(parents=True)
    upstream.write_text("task: detection\n")
    dataset = tmp_path / "dataset"
    for path in (dataset / "images/train", dataset / "images/val", dataset / "annotations"):
        path.mkdir(parents=True, exist_ok=True)
    (dataset / "annotations/instances_train.json").write_text("{}")
    (dataset / "annotations/instances_val.json").write_text("{}")
    (dataset / "manifest.json").write_text(json.dumps({
        "representation": "rgb_coco_labels_clean_v1", "train_count": 1600, "val_count": 400,
        "labels_identity": {"aggregate_sha256": "6a670b95b33e803e5d25fc30d7bbd7985cbc4799234c37ff42b3b1c9204025a4"},
    }))
    backbone = tmp_path / "vitt_distill.pt"; backbone.write_bytes(b"fixture")
    checkpoint = tmp_path / "deimv2_s.pth"; checkpoint.write_bytes(b"fixture")
    monkeypatch.setattr(RUNTIME, "verify_git_checkout", lambda *_: RUNTIME.DEIMV2_COMMIT)
    monkeypatch.setattr(RUNTIME, "verify_file", lambda *_: "ok")
    config = RUNTIME.build_runtime_config(deim, dataset, backbone, checkpoint, "formal")
    assert config["num_classes"] == 12 and config["remap_mscoco_category"] is False
    assert config["epoches"] == 72
    assert config["train_dataloader"]["dataset"]["ann_file"].endswith("instances_train.json")
    assert config["DINOv3STAs"]["weights_path"].endswith("vitt_distill.pt")


def test_ir_decision_requires_all_predeclared_checks():
    passing = []
    for index in range(50):
        passing.append({
            "stem": "s%03d" % (index // 2), "alignment_reliable": True, "class_id": 0,
            "heat_snr": 1.2, "raw_ir_snr": 0.8,
        })
    report = IR.summarize(passing, image_count=100, reliable_image_count=25)
    assert report["decision"] == "PASS"
    passing[0]["heat_snr"] = -100.0
    passing[0]["raw_ir_snr"] = -100.0
    report = IR.summarize(passing, image_count=100, reliable_image_count=19)
    assert report["decision"] == "REJECT"


def test_ir_box_contrast_detects_hot_center():
    image = np.zeros((40, 40), dtype=np.float32)
    image[15:25, 15:25] = 1.0
    metric = IR.box_contrast(image, np.ones_like(image, dtype=bool), (0, 0.5, 0.5, 0.25, 0.25))
    assert metric is not None and metric["snr"] > 10


def test_ir_alignment_applies_inverse_reported_displacement():
    image = np.zeros((40, 50), dtype=np.uint8)
    image[12:24, 15:28] = 255
    shifted = cv2.warpAffine(image, np.float32([[1, 0, 4], [0, 1, -3]]), (50, 40))
    aligned, _ = IR.align_ir(shifted, np.ones_like(shifted, dtype=bool), dx=4, dy=-3)
    assert np.array_equal(aligned, image)
