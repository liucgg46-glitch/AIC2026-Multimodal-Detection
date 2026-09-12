import ast
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict

import torch


REPO_ROOT = Path(__file__).parents[1]
SCRIPT = REPO_ROOT / "scripts" / "data" / "validate_depth_c2_c4.py"
SPEC = importlib.util.spec_from_file_location("validate_depth_c2_c4", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def smoke_payload() -> Dict[str, object]:
    loader = {
        "batch_obtained": True,
        "dataset_size": 2,
        "image_shape": [2, 3, 4, 5],
        "image_dtype": "torch.uint8",
        "configured_batch_size": 2,
        "actual_sample_count": 2,
        "finite": True,
        "image_min": 0.0,
        "image_max": 255.0,
        "preprocessing_stage": "test",
        "class_tensor_present": True,
        "class_tensor_shape": [1, 1],
        "classes_read": True,
        "bbox_tensor_present": True,
        "bbox_tensor_shape": [1, 4],
        "bboxes_read": True,
    }
    return {
        "schema_version": 2,
        "command": MODULE.COMMAND,
        "git_head_sha": "a" * 40,
        "environment": {
            "python_version": "3.10.0",
            "pytorch_version": "test",
            "ultralytics_version": "test",
            "python38_static_compatibility": "passed",
        },
        "source_contract": {},
        "candidates": [
            {
                "candidate": "validmask",
                "manifest_sha256": "b" * 64,
                "train_loader_smoke": loader,
                "val_loader_smoke": loader,
            }
        ],
        "overall_status": "PASS",
    }


def test_validation_script_parses_as_python38() -> None:
    ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT), feature_version=(3, 8))


def test_index_files_counts_casefold_duplicate_extensions(tmp_path: Path) -> None:
    (tmp_path / "sample.png").write_bytes(b"png")
    (tmp_path / "sample.JPG").write_bytes(b"jpg")
    (tmp_path / "ignored.cache").write_bytes(b"cache")
    index, count = MODULE.index_files(tmp_path, {".png", ".jpg"})
    assert count == 2
    assert set(index) == {"sample"}
    assert MODULE.duplicate_count(index) == 1


def test_summarize_batch_records_actual_loader_values() -> None:
    image = torch.tensor(
        [[[[0, 255], [1, 2]]] * 3, [[[3, 4], [5, 6]]] * 3], dtype=torch.uint8
    )
    batch = {
        "img": image,
        "cls": torch.tensor([[2.0]], dtype=torch.float32),
        "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.2]], dtype=torch.float32),
    }
    result = MODULE.summarize_batch(batch, dataset_size=1600)
    assert result["dataset_size"] == 1600
    assert result["actual_sample_count"] == 2
    assert result["image_shape"] == [2, 3, 2, 2]
    assert result["image_dtype"] == "torch.uint8"
    assert result["image_min"] == 0.0 and result["image_max"] == 255.0
    assert result["class_tensor_shape"] == [1, 1]
    assert result["bbox_tensor_shape"] == [1, 4]


def test_write_outputs_generates_json_and_compact_log(
    tmp_path: Path, monkeypatch
) -> None:
    json_output = tmp_path / "evidence.json"
    log_output = tmp_path / "evidence.log"
    monkeypatch.setattr(MODULE, "JSON_OUTPUT", json_output)
    monkeypatch.setattr(MODULE, "LOG_OUTPUT", log_output)
    payload = smoke_payload()
    MODULE.write_outputs(payload)
    assert json.loads(json_output.read_text(encoding="utf-8"))["git_head_sha"] == "a" * 40
    log = log_output.read_text(encoding="utf-8")
    assert "Git HEAD: {}".format("a" * 40) in log
    assert "actual_samples=2" in log
    assert log.endswith("OVERALL: PASS\n")


def test_repository_git_head_is_full_sha() -> None:
    assert len(MODULE.git_head_sha()) == 40
