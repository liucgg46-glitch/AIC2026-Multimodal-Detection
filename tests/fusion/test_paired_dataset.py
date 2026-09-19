import random
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.fusion.paired_dataset import (PairedDataset, canonical_records, pair_records,
                                     DISABLED, check_augmentation, read_raw3)
from scripts.data import prepare_ir_yolo as canonical


@pytest.fixture
def fixture_data(tmp_path):
    for name in ("rgb", "ir", "labels", "ir_labels"):
        (tmp_path / name).mkdir()
    yy, xx = np.mgrid[:37, :59]
    image = np.stack((xx * 3, yy * 5, (xx + yy) * 2), axis=-1).astype(np.uint8)
    for stem in ("a", "b", "c"):
        for modality in ("rgb", "ir"):
            cv2.imwrite(str(tmp_path / modality / f"{stem}.png"), image)
        for labels in ("labels", "ir_labels"):
            (tmp_path / labels / f"{stem}.txt").write_text("0 0.3 0.4 0.2 0.2\n", encoding="utf-8")
    (tmp_path / "train.txt").write_text("a\nb\n", encoding="utf-8")
    (tmp_path / "val.txt").write_text("c\n", encoding="utf-8")
    return tmp_path


def pairs(root):
    return pair_records(root / "rgb", root / "ir", root / "labels",
                        {m: root / f"{m}.txt" for m in ("train", "val")}, (2, 1), root / "ir_labels")


def test_actual_canonical_pairing():
    records = canonical_records()
    assert [len(records[m]) for m in ("train", "val")] == [1600, 400]
    root = Path(__file__).resolve().parents[2]
    for mode, digest in (("train", canonical.CANONICAL_TRAIN_NORMALIZED_SHA256),
                         ("val", canonical.CANONICAL_VAL_NORMALIZED_SHA256)):
        split = root / f"data/splits/{mode}.txt"
        assert canonical.normalized_split_sha256(split) == digest
        assert [r[0] for r in records[mode]] == canonical.read_stems(split, len(records[mode]))
        for stem, rgb, ir, label in records[mode]:
            assert rgb.stem == ir.stem == label.stem == stem


@pytest.mark.parametrize("fault", ["missing", "duplicate", "case_ambiguous", "split", "overlap", "label", "label_missing"])
def test_pairing_fail_fast(fixture_data, fault):
    root = fixture_data
    if fault == "missing":
        (root / "ir/a.png").rename(root / "ir/other.png")
    elif fault in ("duplicate", "case_ambiguous"):
        name = "a.jpg" if fault == "duplicate" else "A.jpg"
        (root / "ir" / name).write_bytes((root / "ir/a.png").read_bytes())
    elif fault == "split":
        (root / "train.txt").write_text("a\na\n", encoding="utf-8")
    elif fault == "overlap":
        (root / "val.txt").write_text("a\n", encoding="utf-8")
    elif fault == "label":
        (root / "ir_labels/a.txt").write_text("1 0.3 0.4 0.2 0.2\n", encoding="utf-8")
    else:
        (root / "ir_labels/a.txt").rename(root / "ir_labels/z.txt")
    with pytest.raises(ValueError):
        pairs(root)


def test_sample_and_batch_order(fixture_data, hyp):
    root = fixture_data
    bgr = np.full((37, 59, 3), (10, 20, 30), dtype=np.uint8)
    ir = np.full((37, 59, 3), (40, 50, 60), dtype=np.uint8)
    cv2.imwrite(str(root / "rgb/a.png"), bgr)
    cv2.imwrite(str(root / "ir/a.png"), ir)
    records = pairs(root)["train"]
    dataset = PairedDataset(records, 64, hyp)
    sample = dataset[0]
    assert read_raw3(records[0][1]).shape == read_raw3(records[0][2]).shape == (37, 59, 3)
    assert (root / "labels/a.txt").read_bytes() == (root / "ir_labels/a.txt").read_bytes()
    assert sample["img"][:, 32, 32].tolist() == [30, 20, 10, 60, 50, 40]
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)))
    assert batch["img"].shape == (2, 6, 64, 64)
    assert batch["batch_idx"].tolist() == [0, 1]


@pytest.mark.parametrize("vertical,horizontal", [(0., 0.), (1., 0.), (0., 1.), (1., 1.), (0.5, 0.5)])
def test_synchronized_geometry_and_boxes(fixture_data, hyp, vertical, horizontal):
    hyp.flipud, hyp.fliplr = vertical, horizontal
    dataset = PairedDataset(pairs(fixture_data)["train"], 64, hyp, augment=True)
    random.seed(2026)
    np.random.seed(2026)
    sample = dataset[0]
    assert torch.equal(sample["img"][:3], sample["img"][3:])
    random.seed(2026)
    np.random.seed(2026)
    replay = dataset[0]
    assert torch.equal(sample["img"], replay["img"])
    assert torch.equal(sample["bboxes"], replay["bboxes"])
    if vertical != 0.5:
        ratio = 64 / 59
        cy = (0.4 * 37 * ratio + 12) / 64
        expected = [0.7 if horizontal else 0.3, 1 - cy if vertical else cy, 0.2, 0.2 * 37 * ratio / 64]
        np.testing.assert_allclose(sample["bboxes"][0], expected, atol=1e-6)


def test_rgb_hsv_never_changes_ir(fixture_data, hyp):
    hyp.flipud = hyp.fliplr = 0.0
    plain = PairedDataset(pairs(fixture_data)["train"], 64, hyp, augment=True)[0]
    hyp.hsv_h, hyp.hsv_s, hyp.hsv_v = 0.4, 0.8, 0.8
    np.random.seed(2026)
    colored = PairedDataset(pairs(fixture_data)["train"], 64, hyp, augment=True)[0]
    assert torch.equal(plain["img"][3:], colored["img"][3:])
    assert not torch.equal(plain["img"][:3], colored["img"][:3])
    assert torch.equal(plain["bboxes"], colored["bboxes"])


@pytest.mark.parametrize("key", DISABLED)
def test_unsafe_augmentation_rejected(hyp, key):
    setattr(hyp, key, 0.1)
    with pytest.raises(ValueError, match=key):
        check_augmentation(hyp)


def test_shape_mismatch_rejected(fixture_data, hyp):
    cv2.imwrite(str(fixture_data / "ir/a.png"), np.zeros((32, 32, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="shape mismatch"):
        PairedDataset(pairs(fixture_data)["train"], 64, hyp)
