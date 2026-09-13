"""Exercise inference staging with real C4 PNG encoding and synthetic files only."""
import ast
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("depth_inference", ROOT / "scripts/data/prepare_depth_inference_yolo.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture(params=["data/raw/test/depth", "data/raw/prelim_test/depth"])
def project(tmp_path, monkeypatch, request):
    monkeypatch.setattr(MODULE, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "EXPECTED_COUNT", 2)
    source, output = tmp_path / request.param, tmp_path / MODULE.OUTPUT
    source.mkdir(parents=True)
    depth = np.array([[0, 1, 300, 1000, 19999, 20000, 65535]], dtype=np.uint16)
    assert cv2.imwrite(str(source / "first.PNG"), depth)
    assert cv2.imwrite(str(source / "second.JPG"), np.full((2, 3, 3), 120, dtype=np.uint8))
    def forbidden(*args, **kwargs):
        pytest.fail("Inference must not read split/labels or fit statistics")
    for name in ("validate_sources", "index_labels", "read_stems"):
        monkeypatch.setattr(MODULE.c4.c2, name, forbidden)
    monkeypatch.setattr(MODULE.c4, "fit_frozen_percentile_params", forbidden)
    monkeypatch.setattr(MODULE.c4.c3, "fit_train_png_percentiles", forbidden)
    return source, output


def test_real_writer_metadata_isolation_and_source_unchanged(project):
    source, output = project
    before = MODULE.source_inventory(source)
    manifest = MODULE.build_view(source, output)
    result = cv2.imread(str(output / "images/first.png"), cv2.IMREAD_UNCHANGED)
    assert result.dtype == np.uint8 and result.shape == (1, 7, 3)
    assert result[0, [0, 2, 5, 6], 0].tolist() == [0, 255, 1, 1]
    assert (result[0, 1:] >= 1).all()
    assert (output / "images/second.jpg").read_bytes() == (source / "second.JPG").read_bytes()
    assert MODULE.source_inventory(source) == before
    assert manifest["source_path"] == source.relative_to(MODULE.PROJECT_ROOT).as_posix()
    assert all(r["source"] == manifest["source_path"] + "/" + Path(r["source"]).name
               for r in manifest["files"])
    metadata = manifest["conversion"]
    assert (metadata["near_clip_mm"], metadata["far_clip_mm"], metadata["invalid_value"]) == (300, 19999, 0)
    assert metadata["valid_output_range"] == [1, 255]
    assert manifest["total"] == manifest["unique_casefold_stems"] == 2
    assert manifest["png_count"] == manifest["jpg_count"] == 1
    assert manifest["source_unchanged"] and manifest["jpg_physical_unit"] == "unknown"
    first = (output / "manifest.json").read_bytes()
    # Move the completed test view aside; production offers no force/delete operation.
    output.rename(output.with_name("test_previous"))
    MODULE.build_view(source, output)
    assert first == (output / "manifest.json").read_bytes()
    (output / "images/second.jpg").write_bytes(b"downstream change")
    assert MODULE.source_inventory(source) == before


@pytest.mark.parametrize("bad", ["duplicate", "casefold", "count", "unsupported", "existing", "bad_png"])
def test_failures(project, monkeypatch, bad):
    source, output = project
    if bad in {"duplicate", "casefold"}:
        (source / "second.JPG").rename(source / ("first.jpg" if bad == "duplicate" else "FIRST.jpg"))
    elif bad == "count":
        monkeypatch.setattr(MODULE, "EXPECTED_COUNT", 1000)
    elif bad == "unsupported":
        (source / "unknown.bmp").write_bytes(b"bad")
    elif bad == "existing":
        output.mkdir(parents=True)
        (output / "keep").write_bytes(b"sentinel")
    else:
        assert cv2.imwrite(str(source / "first.PNG"), np.zeros((2, 2), dtype=np.uint8))
    with pytest.raises(ValueError):
        MODULE.build_view(source, output)
    if bad == "existing":
        assert (output / "keep").read_bytes() == b"sentinel"


@pytest.mark.parametrize("path", ["data/raw/test/depth/out", "data/processed/depth_trainable/inverse", "data/processed/depth_inference"])
def test_output_boundary(project, path):
    source, _ = project
    with pytest.raises(ValueError, match="canonical"):
        MODULE.build_view(source, MODULE.PROJECT_ROOT / path)


def test_parent_redirection_rejected(project, monkeypatch):
    source, output = project
    original = Path.resolve
    processed = MODULE.PROJECT_ROOT / "data/processed"
    def redirected(path, *args, **kwargs):
        if path == processed:
            return MODULE.PROJECT_ROOT.parent / "outside"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", redirected)
    with pytest.raises(ValueError, match="Redirected"):
        MODULE.build_view(source, output)
    assert not output.exists()


def test_python38_syntax_and_production_count():
    assert MODULE.EXPECTED_COUNT == 1000
    ast.parse(Path(MODULE.__file__).read_text(encoding="utf-8"), feature_version=(3, 8))


@pytest.mark.parametrize("arbitrary", ["data/raw/train/depth", "data/raw/other/depth", "unrelated"])
def test_arbitrary_source_rejected(project, arbitrary):
    _, output = project
    source = MODULE.PROJECT_ROOT / arbitrary
    source.mkdir(parents=True)
    with pytest.raises(ValueError, match="Source must be canonical"):
        MODULE.build_view(source, output)
    assert not output.exists()


def test_alias_uses_same_production_api(project, monkeypatch):
    from unittest.mock import Mock
    source, output = project
    writer = Mock(wraps=MODULE.c4.write_candidate_png)
    copy = Mock(wraps=MODULE.c4.copy_isolated)
    metadata = Mock(wraps=MODULE.c4.candidate_conversion_metadata)
    monkeypatch.setattr(MODULE.c4, "write_candidate_png", writer)
    monkeypatch.setattr(MODULE.c4, "copy_isolated", copy)
    monkeypatch.setattr(MODULE.c4, "candidate_conversion_metadata", metadata)
    MODULE.build_view(source, output)
    assert writer.call_count == copy.call_count == 1
    assert writer.call_args.args[0] == source / "first.PNG"
    assert writer.call_args.args[2:] == ("inverse", None)
    assert copy.call_args.args[0] == source / "second.JPG"
    metadata.assert_called_once_with("inverse", None)
