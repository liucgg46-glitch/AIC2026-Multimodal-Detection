import ast
import importlib.util
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("s003_submission", ROOT / "scripts/inference/prepare_s003_submission.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "EXPECTED_COUNT", 2)
    images, txt = tmp_path / "images", tmp_path / "txt"
    images.mkdir()
    txt.mkdir()
    for name in ("a", "b"):
        (images / (name + ".png")).write_bytes(b"stem fixture")
    (txt / "a.txt").write_text("0 0.5 0.5 0.1 0.1 0.9\n" * 100, encoding="utf-8")
    (txt / "b.txt").write_bytes(b"")
    return images, txt


def test_statistics_and_synthetic_zip(predictions, tmp_path):
    images, txt = predictions
    archive = tmp_path / "synthetic.zip"
    result = MODULE.prepare(images, txt, archive)
    assert result["nonempty"] == result["empty"] == result["hit_max_det_100"] == 1
    assert result["total_boxes"] == result["max_boxes_per_image"] == 100
    assert result["mean_boxes_per_image"] == 50
    assert result["missing"] == result["extra"] == result["nan_inf_count"] == 0
    assert len(result["zip_sha256"]) == 64
    with zipfile.ZipFile(archive) as z:
        assert z.namelist() == ["a.txt", "b.txt"]
        assert z.read("a.txt") == (txt / "a.txt").read_bytes()
    with pytest.raises(FileExistsError):
        MODULE.prepare(images, txt, archive)


@pytest.mark.parametrize("bad", ["missing", "extra", "non_txt", "class", "xywh", "confidence", "nan", "inf", "count"])
def test_rejects_invalid_submission(predictions, monkeypatch, bad):
    images, txt = predictions
    rows = {"class": "12 0.5 0.5 0.1 0.1 0.9", "xywh": "0 2 0.5 0.1 0.1 0.9",
            "confidence": "0 0.5 0.5 0.1 0.1 2", "nan": "0 nan 0.5 0.1 0.1 0.9",
            "inf": "0 0.5 0.5 0.1 0.1 inf"}
    if bad in rows:
        (txt / "a.txt").write_text(rows[bad], encoding="utf-8")
    elif bad == "missing":
        (txt / "a.txt").unlink()
    elif bad == "extra":
        (txt / "c.txt").write_bytes(b"")
    elif bad == "non_txt":
        (txt / "manifest.json").write_bytes(b"{}")
    else:
        monkeypatch.setattr(MODULE, "EXPECTED_COUNT", 1000)
    with pytest.raises(ValueError):
        MODULE.prepare(images, txt)


def test_python38_syntax():
    ast.parse(Path(MODULE.__file__).read_text(encoding="utf-8"), feature_version=(3, 8))
