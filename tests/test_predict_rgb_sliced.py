"""Geometry and baseline-preservation checks for R7 sliced inference."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

import scripts.inference.predict_rgb_sliced as SLICED
from scripts.inference.predict_rgb_sliced import (
    convert_tile_box, format_rows, merge_with_baseline, tile_windows,
)


def test_four_overlapping_windows_and_small_image_skip():
    assert tile_windows(640, 360) == []
    assert tile_windows(1920, 1080) == [
        (0, 0, 1152, 648), (768, 0, 1920, 648),
        (0, 432, 1152, 1080), (768, 432, 1920, 1080),
    ]


def test_tile_box_projection_and_internal_edge_filter():
    window = (768, 432, 1920, 1080)
    projected = convert_tile_box([100, 100, 200, 200], window, 1920, 1080, 5, 0.9)
    assert projected is not None
    assert projected[0] == 5
    assert projected[1:5] == [918 / 1920, 582 / 1080, 100 / 1920, 100 / 1080]
    assert projected[5] == 0.9 * 0.8
    assert convert_tile_box([0, 50, 20, 70], window, 1920, 1080, 5, 0.9) is None
    # The same edge is the actual image boundary on the left-most tile.
    assert convert_tile_box([0, 50, 20, 70], (0, 0, 1152, 648), 1920, 1080, 5, 0.9)


def test_merge_keeps_full_image_boxes_and_adds_only_distinct_tiles():
    base = [[0, 0.5, 0.5, 0.1, 0.1, 0.6]]
    tiles = [
        [0, 0.5, 0.5, 0.1, 0.1, 0.72],  # duplicate, even with higher confidence
        [0, 0.1, 0.1, 0.02, 0.02, 0.5],
        [0, 0.1, 0.1, 0.02, 0.02, 0.4],  # tile duplicate
    ]
    merged = merge_with_baseline(base, tiles)
    assert len(merged) == 2
    assert base[0] in merged
    assert tiles[1] in merged
    assert len(format_rows(merged).decode("utf-8").splitlines()) == 2


def test_run_projects_model_output_and_writes_submission_rows(tmp_path, monkeypatch):
    images = tmp_path / "images"
    baseline = tmp_path / "baseline"
    images.mkdir()
    baseline.mkdir()
    cv2.imwrite(str(images / "sample.jpg"), np.zeros((1080, 1920, 3), dtype=np.uint8))
    (baseline / "sample.txt").write_bytes(b"")
    calls = []

    class FakeModel:
        def predict(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                boxes = SimpleNamespace(
                    xyxy=torch.tensor([[100., 100., 200., 200.]]),
                    cls=torch.tensor([5.]),
                    conf=torch.tensor([0.9]),
                )
            else:
                boxes = None
            return [SimpleNamespace(boxes=boxes)]

    monkeypatch.setattr(SLICED, "YOLO", lambda path: FakeModel())
    output = tmp_path / "merged"
    report = SLICED.run(tmp_path / "best.pt", images, baseline, output, "cpu", False)
    assert report["images"] == report["tiled_images"] == 1
    assert report["tile_candidates_before_merge"] == 1
    assert len(calls) == 4
    assert calls[0]["source"].shape == (648, 1152, 3)
    fields = (output / "sample.txt").read_text().split()
    assert fields[0] == "5"
    assert float(fields[-1]) == pytest.approx(0.72)
