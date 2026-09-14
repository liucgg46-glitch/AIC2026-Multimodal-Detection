import ast
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from scripts.analysis.audit_f001_alignment import (
    draw_normalized_gt,
    render_review_image,
    run_alignment_audit,
    select_review_rows,
)
from src.fusion import diagnostics
from src.fusion.diagnostics import (
    ABLATION_MODES,
    AblationDataset,
    ablation_context,
    collect_fusion_statistics,
    compare_ctrl001_to_f001,
    deterministic_ir_shuffle,
    immutable_checkpoint,
    run_fresh_model_plan,
    sha256,
)
from src.fusion.fusion_modules import GatedResidualFusion


ROOT = Path(__file__).resolve().parents[2]


def write_image(path, value=100, shape=(12, 20, 3)):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full(shape, value, dtype=np.uint8))
    return path


def test_residual_off_returns_exact_rgb():
    model = SimpleNamespace(
        fusion4=GatedResidualFusion(4, zero_init=False),
        fusion5=GatedResidualFusion(8, zero_init=False),
    )
    for fusion in (model.fusion4, model.fusion5):
        rgb = torch.randn(2, fusion.proj.in_channels, 5, 7)
        ir = torch.randn_like(rgb)
        with ablation_context(model, "RESIDUAL_OFF"):
            torch.testing.assert_close(fusion(rgb, ir), rgb, rtol=0, atol=0)
        assert not torch.equal(fusion(rgb, ir), rgb)


class TinyBaseDataset:
    labels = [{"cls": torch.zeros(0, 1)}]
    records = [("sample", None, None, None)]
    im_files = ["sample"]

    def __len__(self):
        return 1

    def __getitem__(self, index):
        del index
        return {"img": torch.cat((torch.ones(3, 4, 5), torch.full((3, 4, 5), 9)), 0)}


def test_ir_zero_preserves_rgb_transport_channels():
    original = TinyBaseDataset()[0]["img"]
    zeroed = AblationDataset(TinyBaseDataset(), "IR_ZERO")[0]["img"]
    torch.testing.assert_close(zeroed[:3], original[:3], rtol=0, atol=0)
    assert torch.count_nonzero(zeroed[3:]) == 0
    assert torch.count_nonzero(original[3:]) > 0


def test_ir_shuffle_is_deterministic_deranged_and_shape_safe(tmp_path):
    records = []
    for index, shape in enumerate(((8, 12, 3), (8, 12, 3), (10, 16, 3), (10, 16, 3))):
        stem = "s%d" % index
        rgb = write_image(tmp_path / (stem + "_rgb.png"), index + 1, shape)
        ir = write_image(tmp_path / (stem + "_ir.png"), index + 10, shape)
        label = tmp_path / (stem + ".txt")
        label.write_text("", encoding="utf-8")
        records.append((stem, rgb, ir, label))
    first, manifest1 = deterministic_ir_shuffle(records, 2026)
    second, manifest2 = deterministic_ir_shuffle(records, 2026)
    assert manifest1 == manifest2
    assert [record[2] for record in first] == [record[2] for record in second]
    assert all(row["rgb_stem"] != row["shuffled_ir_stem"] for row in manifest1)
    assert all(cv2.imread(str(record[1])).shape == cv2.imread(str(record[2])).shape for record in first)
    for height, width in {(row["height"], row["width"]) for row in manifest1}:
        group = [row for row in manifest1 if (row["height"], row["width"]) == (height, width)]
        assert sorted(row["original_ir_stem"] for row in group) == sorted(row["shuffled_ir_stem"] for row in group)
        assert len({row["shuffled_ir_stem"] for row in group}) == len(group)


def test_ablation_modes_and_statistics_receive_distinct_fresh_models_and_order_is_invariant():
    loaded = []

    def loader():
        model = SimpleNamespace(mutated=False)
        loaded.append(model)
        return model

    def mode_runner(model, mode):
        assert model.mutated is False
        model.mutated = True
        return {"mode": mode, "contract": mode.lower()}

    def statistics_runner(model):
        assert model.mutated is False
        model.mutated = True
        return {"contract": "statistics"}

    def execute(order):
        loaded.clear()
        reports, statistics = run_fresh_model_plan(
            order, loader, mode_runner, statistics_runner,
        )
        assert len(loaded) == len(order) + 1
        assert len({id(model) for model in loaded}) == len(loaded)
        return {row["mode"]: row["contract"] for row in reports}, statistics

    forward = execute(ABLATION_MODES)
    reverse = execute(tuple(reversed(ABLATION_MODES)))
    assert forward == reverse


def test_fixed_val_records_requires_exact_ordered_400(monkeypatch, tmp_path):
    stems = ["v%03d" % index for index in range(400)]
    split = tmp_path / "data/splits/val.txt"
    split.parent.mkdir(parents=True)
    split.write_text("\n".join(stems) + "\n", encoding="utf-8")
    records = [(stem, Path(stem), Path(stem), Path(stem)) for stem in stems]
    monkeypatch.setattr(diagnostics, "ROOT", tmp_path)
    monkeypatch.setattr(diagnostics, "canonical_records", lambda: {"val": records})
    assert [record[0] for record in diagnostics.fixed_val_records()] == stems


def test_all_ablation_modes_leave_checkpoint_bytes_unchanged(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"immutable F001 checkpoint")
    before = sha256(checkpoint)
    model = SimpleNamespace(
        fusion4=GatedResidualFusion(2, zero_init=False),
        fusion5=GatedResidualFusion(2, zero_init=False),
    )
    with immutable_checkpoint(checkpoint, before):
        for mode in ABLATION_MODES:
            with ablation_context(model, mode):
                model.fusion4(torch.ones(1, 2, 2, 2), torch.ones(1, 2, 2, 2))
    assert sha256(checkpoint) == before


def test_checkpoint_guard_still_verifies_on_diagnostic_exception(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"before")
    with pytest.raises(RuntimeError, match="changed"):
        with immutable_checkpoint(checkpoint, sha256(checkpoint)):
            checkpoint.write_bytes(b"after")
            raise ValueError("diagnostic failed")


class StatsDataset:
    collate_fn = staticmethod(lambda rows: {"img": torch.stack([row["img"] for row in rows])})

    def __init__(self):
        self.records = [("v%03d" % index, None, None, None) for index in range(400)]

    def __len__(self):
        return 400

    def __getitem__(self, index):
        return {"img": torch.full((6, 8, 8), index % 251, dtype=torch.uint8)}


class StatsModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.fusion4 = GatedResidualFusion(3, zero_init=False)
        self.fusion5 = GatedResidualFusion(3, zero_init=False)

    def backbone_features(self, images):
        rgb, ir = images[:, :3] + self.anchor, images[:, 3:6] + self.anchor
        return {6: rgb, 10: rgb[:, :, ::2, ::2]}, {6: ir, 10: ir[:, :, ::2, ::2]}


def test_fusion_statistics_traverses_all_val400_samples():
    report = collect_fusion_statistics(
        StatsModel().eval(), StatsDataset(), SimpleNamespace(batch=37, workers=0), "checkpoint", "git"
    )
    assert report["sample_count"] == 400
    for level in ("P4", "P5"):
        assert len(report["levels"][level]["per_image"]) == 400
        assert report["levels"][level]["gate"]["p50"] is not None


def _registration(stem, visible, infrared):
    del visible, infrared
    index = int(stem[1:])
    if index < 200:
        return {
            "reliable": True, "reason": "reliable", "dx": float(index % 5), "dy": 1.0,
            "magnitude": math_hypot(float(index % 5), 1.0), "confidence": 0.8,
            "edge_correlation": 0.4, "peak_margin": 0.02, "peak_ratio": 1.3,
        }
    return {
        "reliable": False, "reason": "low_edge_correlation,ambiguous_peak", "confidence": 0.0,
        "edge_correlation": 0.1, "peak_margin": 0.0, "peak_ratio": 1.0,
    }


def math_hypot(x, y):
    return float((x * x + y * y) ** 0.5)


def test_alignment_audit_covers_all_400_and_separates_png_jpg(tmp_path):
    png = write_image(tmp_path / "shared.png")
    jpg = write_image(tmp_path / "shared.jpg")
    label = tmp_path / "label.txt"
    label.write_text("0 0.5 0.5 0.4 0.2\n", encoding="utf-8")
    records = [
        ("v%03d" % index, png if index % 2 == 0 else jpg, png if index % 2 == 0 else jpg, label)
        for index in range(400)
    ]
    before = label.read_bytes()
    summary = run_alignment_audit(records, tmp_path / "audit", result_provider=_registration, render=False)
    assert summary["attempted"] == 400
    assert summary["fixed_val_count"] == 400
    assert summary["reliable"] == 200
    assert summary["reliable_ratio_by_encoding"]["rgb"]["PNG"]["attempted"] == 200
    assert summary["reliable_ratio_by_encoding"]["rgb"]["JPG"]["attempted"] == 200
    assert summary["reliable_ratio_by_encoding"]["ir"]["PNG"]["attempted"] == 200
    header = (tmp_path / "audit/registration.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "rgb_encoding" in header and "ir_encoding" in header
    assert "not an empirical error distribution for all val400" in summary["reliable_subset_scope_note"]
    assert len((tmp_path / "audit/registration.csv").read_text(encoding="utf-8").splitlines()) == 401
    assert label.read_bytes() == before


def test_ir_review_uses_same_visible_boxes_without_modifying_label(tmp_path):
    image = np.full((40, 80, 3), 100, dtype=np.uint8)
    boxes = [(0, 0.5, 0.5, 0.4, 0.2)]
    torch.testing.assert_close(
        torch.from_numpy(draw_normalized_gt(image, boxes)),
        torch.from_numpy(draw_normalized_gt(image.copy(), boxes)),
    )
    rgb, ir = write_image(tmp_path / "one_rgb.png", shape=image.shape), write_image(tmp_path / "one_ir.png", shape=image.shape)
    label = tmp_path / "one.txt"
    label.write_text("0 0.5 0.5 0.4 0.2\n", encoding="utf-8")
    before = label.read_bytes()
    render_review_image(
        ("one", rgb, ir, label),
        {"reliable": True, "dx": 1.0, "dy": 2.0, "magnitude": 2.236, "confidence": 0.8, "reason": "reliable"},
        tmp_path / "review.png",
    )
    assert (tmp_path / "review.png").is_file()
    assert label.read_bytes() == before


def test_manual_review_selects_three_deterministic_examples_per_available_category():
    rows = []
    reasons = ("search_boundary", "low_edge_correlation", "ambiguous_peak")
    for index in range(30):
        reliable = index < 12
        rows.append({
            "stem": "s%02d" % index,
            "reliable": reliable,
            "magnitude": float(index) if reliable else None,
            "reason": "reliable" if reliable else reasons[(index - 12) % len(reasons)],
            "ir_fov_ratio": 0.5 + index / 100.0,
        })
    selected = select_review_rows(rows)
    for category in ("reliable_small_displacement", "reliable_large_displacement", "unreliable",
                     "search_boundary", "low_edge_correlation", "ambiguous_peak", "worst_ir_fov"):
        assert sum(category in row["selection_reasons"].split(",") for row in selected) >= 3


def test_ctrl001_matches_f001_except_machine_approved_rgb_fields():
    ctrl = ROOT / "configs/experiments/CTRL001_RGB_F001_AUG.yaml"
    f001 = ROOT / "configs/experiments/F001_RGB_IR_GATED_P45_YOLO11N.yaml"
    report = compare_ctrl001_to_f001(ctrl, f001)
    assert report["passed"] is True
    assert report["augmentation_equal"] is True
    assert report["mismatches"] == {}
    assert all(report["control_contract"].values())


def test_new_diagnostics_are_python38_syntax_compatible():
    paths = (
        ROOT / "src/fusion/control.py",
        ROOT / "src/fusion/diagnostics.py",
        ROOT / "scripts/analysis/analyze_f001_fusion.py",
        ROOT / "scripts/analysis/audit_f001_alignment.py",
        ROOT / "scripts/analysis/check_ctrl001_config.py",
        ROOT / "scripts/train/train_ctrl001.py",
    )
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 8))
