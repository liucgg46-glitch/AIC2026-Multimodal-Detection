import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts/data/validate_multimodal_smoke.py"
SPEC = importlib.util.spec_from_file_location("validate_multimodal_smoke", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_yaml(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)
    return path


def valid_yaml():
    return {
        "train": "images/train",
        "val": "images/val",
        "channels": 3,
        "names": {index: "class{}".format(index) for index in range(12)},
    }


@pytest.mark.parametrize("missing", ["train", "val", "names"])
def test_missing_yaml_keys_fail(tmp_path: Path, missing: str) -> None:
    data = valid_yaml()
    data.pop(missing)
    path = write_yaml(tmp_path / "data.yaml", data)
    with pytest.raises(MODULE.SmokeValidationError, match="缺少"):
        MODULE.validate_yaml_contract(path)


def test_missing_data_yaml_fails(tmp_path: Path) -> None:
    with pytest.raises(MODULE.SmokeValidationError, match="不存在"):
        MODULE.validate_yaml_contract(tmp_path / "missing.yaml")


def test_wrong_yaml_channels_fail(tmp_path: Path) -> None:
    data = valid_yaml()
    data["channels"] = 1
    with pytest.raises(MODULE.SmokeValidationError, match="channels"):
        MODULE.validate_yaml_contract(write_yaml(tmp_path / "data.yaml", data))


class Dataset:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


def test_dataset_build_exception_fails() -> None:
    def broken():
        raise ValueError("boom")

    with pytest.raises(MODULE.SmokeValidationError, match="dataset build"):
        MODULE.obtain_first_batch(broken, lambda dataset: [object()])


def test_suppressed_cache_write_preserves_in_memory_version_without_file(tmp_path: Path) -> None:
    cache = {"hash": "abc", "msgs": [], "labels": []}
    cache_path = tmp_path / "labels.cache"
    MODULE.suppress_dataset_cache_write("", cache_path, cache, "1.0")
    assert cache["version"] == "1.0"
    assert not cache_path.exists()


def test_empty_dataset_fails() -> None:
    with pytest.raises(MODULE.SmokeValidationError, match="为空"):
        MODULE.obtain_first_batch(lambda: Dataset(0), lambda dataset: [object()])


def test_loader_without_batch_fails() -> None:
    with pytest.raises(MODULE.SmokeValidationError, match="未返回 batch"):
        MODULE.obtain_first_batch(lambda: Dataset(1), lambda dataset: [])


def valid_batch():
    return {
        "img": torch.zeros((2, 3, 640, 640), dtype=torch.uint8),
        "cls": torch.zeros((2, 1), dtype=torch.float32),
        "bboxes": torch.zeros((2, 4), dtype=torch.float32),
        "batch_idx": torch.tensor([0, 1]),
        "im_file": ["a.png", "b.png"],
    }


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda batch: batch.update(img=torch.zeros((3, 640, 640), dtype=torch.uint8)), "BCHW"),
        (lambda batch: batch.update(img=torch.zeros((1, 3, 640, 640), dtype=torch.uint8)), "batch size"),
        (lambda batch: batch.update(img=torch.zeros((2, 1, 640, 640), dtype=torch.uint8)), "channels"),
        (lambda batch: batch.update(img=torch.full((2, 3, 2, 2), float("nan"))), "dtype|非有限"),
        (lambda batch: batch.pop("cls"), "cls"),
        (lambda batch: batch.update(cls=torch.zeros((2,))), "cls tensor shape"),
        (lambda batch: batch.pop("bboxes"), "bboxes"),
        (lambda batch: batch.update(bboxes=torch.zeros((2, 5))), "bboxes tensor shape"),
    ],
)
def test_malformed_raw_batch_fails(mutation, message) -> None:
    batch = valid_batch()
    mutation(batch)
    with pytest.raises(MODULE.SmokeValidationError, match=message):
        MODULE.summarize_raw_batch(batch, 10)


@pytest.mark.parametrize(
    "image,message",
    [
        (torch.zeros((2, 3, 640, 640), dtype=torch.float64), "float32"),
        (torch.full((2, 3, 2, 2), 1.1, dtype=torch.float32), "range"),
    ],
)
def test_invalid_preprocessed_batch_fails(image, message) -> None:
    with pytest.raises(MODULE.SmokeValidationError, match=message):
        MODULE.summarize_preprocessed_batch({"img": image}, list(image.shape))


def test_forward_exception_is_detectable() -> None:
    class Broken(torch.nn.Module):
        def forward(self, image):
            raise RuntimeError("forward failed")

    with pytest.raises(RuntimeError, match="forward failed"):
        Broken()(torch.zeros((2, 3, 8, 8)))


@pytest.mark.parametrize("output,message", [(None, "None"), ([], "未返回 tensor")])
def test_empty_forward_output_fails(output, message) -> None:
    with pytest.raises(MODULE.SmokeValidationError, match=message):
        MODULE.summarize_forward(output, 2)


def test_nonfinite_forward_output_fails() -> None:
    with pytest.raises(MODULE.SmokeValidationError, match="NaN/Inf"):
        MODULE.summarize_forward(torch.tensor([[float("nan")], [0.0]]), 2)


def test_pretrained_pt_is_rejected() -> None:
    with pytest.raises(MODULE.SmokeValidationError, match="pretrained"):
        MODULE.reject_pretrained_source("yolo11n.pt")


def test_successful_batch_preprocess_and_forward_summaries() -> None:
    raw = valid_batch()
    assert MODULE.summarize_raw_batch(raw, 1600)["image_dtype"] == "torch.uint8"
    processed = MODULE.clone_batch(raw)
    processed["img"] = processed["img"].float() / 255.0
    assert MODULE.summarize_preprocessed_batch(processed, [2, 3, 640, 640])["image_dtype"] == "torch.float32"
    summary = MODULE.summarize_forward((torch.zeros((2, 16, 10)), [torch.ones((2, 8, 5))]), 2)
    assert summary["forward_completed"] is True and summary["tensor_count"] == 2


def test_control_flags_remain_false() -> None:
    payload = MODULE.build_base_payload("a" * 40)
    for key in (
        "training_started",
        "backward_executed",
        "optimizer_step_executed",
        "checkpoint_written",
        "accuracy_evaluated",
        "predict_executed",
    ):
        assert payload[key] is False


def test_deterministic_json_and_no_forbidden_identity(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    payload = {"schema": "test", "path": "data/view/data.yaml", "passed": True}
    MODULE.write_deterministic_json(output, payload)
    first = output.read_bytes()
    MODULE.write_deterministic_json(output, payload)
    assert output.read_bytes() == first
    assert json.loads(first.decode("utf-8")) == payload
    for forbidden in (
        {"timestamp": "now"},
        {"uuid": "value"},
        {"path": "D:/machine/private"},
        {"path": "/home/user/private"},
    ):
        with pytest.raises(MODULE.SmokeValidationError):
            MODULE.write_deterministic_json(output, forbidden)


def test_python38_static_syntax() -> None:
    ast.parse(MODULE_PATH.read_text(encoding="utf-8"), filename=str(MODULE_PATH), feature_version=(3, 8))
    test_path = Path(__file__).resolve()
    ast.parse(test_path.read_text(encoding="utf-8"), filename=str(test_path), feature_version=(3, 8))
