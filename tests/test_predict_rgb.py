from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "inference" / "predict_rgb.py"
SPEC = importlib.util.spec_from_file_location("predict_rgb", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeTensor:
    def __init__(self, values: object) -> None:
        self.values = values

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def tolist(self) -> object:
        return self.values


class FakeBoxes:
    def __init__(
        self,
        coordinates: list[list[float]],
        classes: list[float] | None = None,
        confidences: list[float] | None = None,
    ) -> None:
        count = len(coordinates)
        self.xywhn = FakeTensor(coordinates)
        self.cls = FakeTensor(classes if classes is not None else [0.0] * count)
        self.conf = FakeTensor(confidences if confidences is not None else [0.9] * count)

    def __len__(self) -> int:
        return len(self.xywhn.values)


class FakeResult:
    def __init__(self, path: Path, boxes: FakeBoxes) -> None:
        self.path = str(path)
        self.boxes = boxes


def test_normal_prediction_is_converted_to_submission_line() -> None:
    result = FakeResult(
        Path("sample.jpg"),
        FakeBoxes([[0.5, 0.4, 0.2, 0.1]], classes=[6.0], confidences=[0.75]),
    )

    assert MODULE.prediction_lines(result, 100) == [
        "6 0.50000000 0.40000000 0.20000000 0.10000000 0.75000000"
    ]


@pytest.mark.parametrize(
    "coordinates",
    [
        [0.5, 0.5, 0.0, 0.2],
        [0.5, 0.5, 0.2, 0.0],
    ],
)
def test_zero_area_prediction_is_filtered(coordinates: list[float]) -> None:
    result = FakeResult(Path("sample.jpg"), FakeBoxes([coordinates]))

    assert MODULE.prediction_lines(result, 100) == []


def test_all_degenerate_predictions_allow_empty_output() -> None:
    result = FakeResult(
        Path("sample.jpg"),
        FakeBoxes(
            [
                [0.5, 0.5, 0.0, 0.2],
                [0.5, 0.5, 0.2, 0.0],
            ]
        ),
    )

    assert MODULE.prediction_lines(result, 100) == []


@pytest.mark.parametrize(
    "boxes, expected",
    [
        (FakeBoxes([[0.5, 0.5, 0.2, 0.2]], classes=[12.0]), "class_id 越界"),
        (FakeBoxes([[math.nan, 0.5, 0.2, 0.2]]), "NaN 或 Inf"),
        (FakeBoxes([[0.5, math.inf, 0.2, 0.2]]), "NaN 或 Inf"),
        (FakeBoxes([[1.1, 0.5, 0.2, 0.2]]), "归一化坐标越界"),
        (FakeBoxes([[0.5, 0.5, 0.2, 0.2]], confidences=[1.1]), "confidence 越界"),
    ],
)
def test_invalid_prediction_still_raises(boxes: FakeBoxes, expected: str) -> None:
    result = FakeResult(Path("sample.jpg"), boxes)

    with pytest.raises(MODULE.InferenceError, match=expected):
        MODULE.prediction_lines(result, 100)


def test_max_det_constraint_still_raises() -> None:
    result = FakeResult(
        Path("sample.jpg"),
        FakeBoxes([[0.5, 0.5, 0.2, 0.2]] * 101),
    )

    with pytest.raises(MODULE.InferenceError, match="超过限制 100"):
        MODULE.prediction_lines(result, 100)


def test_inference_uses_directory_source_batch_one_and_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "images"
    source.mkdir()
    image = source / "sample.jpg"
    image.write_bytes(b"image")
    output = tmp_path / "predictions"
    captured: dict[str, object] = {}

    class FakeModel:
        def predict(self, **kwargs: object) -> list[FakeResult]:
            captured.update(kwargs)
            return [
                FakeResult(
                    image,
                    FakeBoxes(
                        [
                            [0.5, 0.5, 0.0, 0.2],
                            [0.5, 0.5, 0.2, 0.0],
                        ]
                    ),
                )
            ]

    monkeypatch.setattr(MODULE, "YOLO", lambda _: FakeModel())

    image_count, prediction_count, _ = MODULE.run_inference(
        "model.pt",
        source,
        output,
        device="cpu",
        imgsz=640,
        conf=0.25,
        iou=0.7,
        max_det=100,
        force=False,
        augment=True,
    )

    assert image_count == 1
    assert prediction_count == 0
    assert (output / "sample.txt").read_text(encoding="utf-8") == ""
    assert captured["source"] == str(source)
    assert not isinstance(captured["source"], list)
    assert captured["batch"] == 1
    assert captured["stream"] is True
    assert captured["augment"] is True
