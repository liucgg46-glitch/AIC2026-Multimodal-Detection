from types import SimpleNamespace

import cv2
import numpy as np
import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import LetterBox

from scripts.inference.predict_quality_fusion import (
    detection_lines, paired_test_records, prepare_image, resolve_device, run_prediction,
)
from scripts.analysis.audit_f002_quality import M960_RECT_VAL_BATCH, validate_mode
from src.fusion.quality_dataset import QualityTriModalDataset, decode_depth, edge_dark_valid_region
from src.fusion.quality_model import QualityTriModalModel, QualityResidual


def make_hyp():
    return SimpleNamespace(
        mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
        degrees=0.0, shear=0.0, perspective=0.0,
        multi_scale=False, bgr=0.0, flipud=0.0, fliplr=0.5,
        translate=0.1, scale=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    )


def write_record(tmp_path, extension):
    rgb_path = tmp_path / ("sample_visible" + extension)
    ir_path = tmp_path / ("sample_infrared" + extension)
    depth_path = tmp_path / ("sample_depth" + extension)
    label_path = tmp_path / "sample.txt"
    rgb = np.full((64, 64, 3), 150, dtype=np.uint8)
    ir = np.full((64, 64, 3), 100, dtype=np.uint8)
    ir[:, :16] = 0
    if extension == ".png":
        depth = np.full((64, 64), 1000, dtype=np.uint16)
        depth[:, :8] = 0
    else:
        depth = np.full((64, 64, 3), 130, dtype=np.uint8)
        depth[:, :8] = 0
    assert cv2.imwrite(str(rgb_path), rgb)
    assert cv2.imwrite(str(ir_path), ir)
    assert cv2.imwrite(str(depth_path), depth)
    label_path.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    return ("sample", rgb_path, ir_path, depth_path, label_path)


def test_depth_branches_have_separate_format_tokens_and_masks(tmp_path):
    png_dir = tmp_path / "png"
    jpg_dir = tmp_path / "jpg"
    png_dir.mkdir()
    jpg_dir.mkdir()
    png = write_record(png_dir, ".png")
    jpg = write_record(jpg_dir, ".jpg")
    png_encoded, png_flag = decode_depth(png[3], (64, 64))
    jpg_encoded, jpg_flag = decode_depth(jpg[3], (64, 64))
    assert not png_flag and jpg_flag
    assert png_encoded[0, 0, 2] == jpg_encoded[0, 0, 2] == 0
    assert png_encoded[0, 20, 2] == jpg_encoded[0, 20, 2] == 255
    assert png_encoded[0, 20, 0] != jpg_encoded[0, 20, 0]


def test_shared_geometric_dataset_keeps_all_transport_channels_and_labels(tmp_path):
    record = write_record(tmp_path, ".png")
    dataset = QualityTriModalDataset([record], 64, make_hyp(), augment=True,
                                    dropout_probability=0.0)
    np.random.seed(3)
    sample = dataset[0]
    assert tuple(sample["img"].shape) == (11, 64, 64)
    assert sample["img"].dtype == torch.uint8
    assert sample["img"][10].max() == 0
    assert sample["cls"].shape[1] == 1
    assert sample["bboxes"].shape[1] == 4
    assert sample["img"][8].max() > 0
    assert sample["img"][9].max() > 0


def test_zero_validity_is_exact_rgb_fallback_even_after_fusion_learns():
    torch.set_num_threads(2)
    rgb = DetectionModel("yolo11m.yaml", ch=3, nc=12, verbose=False).eval()
    tri = QualityTriModalModel(nc=12, verbose=False).eval()
    tri.load_state_dict(rgb.state_dict(), strict=False)
    for group in (tri.ir_fusion, tri.depth_fusion):
        for fusion in group.values():
            with torch.no_grad():
                fusion.proj.weight.normal_()
                fusion.proj.bias.fill_(0.1)
    x = torch.rand(1, 11, 64, 64)
    x[:, 8] = 0
    x[:, 9] = 0
    with torch.inference_mode():
        reference, _ = rgb(x[:, :3])
        fused, _ = tri(x)
    assert torch.equal(reference, fused)


def test_quality_residual_masks_invalid_auxiliary_pixels():
    fusion = QualityResidual(4, 2).eval()
    with torch.no_grad():
        fusion.proj.weight.fill_(1)
        fusion.proj.bias.fill_(1)
    rgb = torch.randn(1, 4, 8, 8)
    aux = torch.ones(1, 2, 8, 8)
    assert torch.equal(fusion(rgb, aux, torch.zeros(1, 1, 8, 8)), rgb)
    valid = torch.ones(1, 1, 8, 8)
    valid[:, :, :4] = 0
    actual = fusion(rgb, aux, valid)
    assert torch.equal(actual[:, :, :4], rgb[:, :, :4])
    assert not torch.equal(actual[:, :, 4:], rgb[:, :, 4:])


def test_frozen_rgb_batchnorm_statistics_remain_fixed_in_train_mode():
    torch.set_num_threads(2)
    model = QualityTriModalModel(nc=12, verbose=False)
    before = model.model[0].bn.running_mean.clone()
    model.train()
    assert not model.model[0].bn.training
    assert model.ir_pyramid.blocks[0][1].training
    image = torch.rand(1, 11, 64, 64)
    model(image)
    assert torch.equal(before, model.model[0].bn.running_mean)


def test_ir_valid_region_excludes_jpeg_dark_edge_without_hiding_interior_dark_pixels():
    ir = np.full((64, 64, 3), 100, dtype=np.uint8)
    ir[:, :12] = 0
    ir[25:35, 25:35] = 0
    valid = edge_dark_valid_region(ir)
    assert valid[30, 5] == 0
    assert valid[30, 30] == 255
    assert valid[30, 45] == 255


def test_quality_prediction_writes_exact_stem_zip_and_valid_six_column_rows(tmp_path):
    record = write_record(tmp_path, ".png")
    rgb_dir, ir_dir, depth_dir = (tmp_path / name for name in ("visible", "infrared", "depth"))
    for directory in (rgb_dir, ir_dir, depth_dir):
        directory.mkdir()
    paths = []
    for source, destination in zip(record[1:4], (rgb_dir, ir_dir, depth_dir)):
        path = destination / "00000001.png"
        path.write_bytes(source.read_bytes())
        paths.append(path)
    paired = paired_test_records(rgb_dir, ir_dir, depth_dir, expected_count=1)
    assert len(paired) == 1
    encoded, original_shape, ratio_pad = prepare_image(*paths, imgsz=64)
    assert encoded.shape == (11, 64, 64)
    assert original_shape == (64, 64)
    assert ratio_pad == ((1.0, 1.0), (0, 0))

    class EmptyModel:
        def __call__(self, tensor):
            return torch.zeros((1, 16, 84), device=tensor.device), None

    output = tmp_path / "submission.zip"
    assert run_prediction(paired, EmptyModel(), output, "cpu", imgsz=64) == 0
    import zipfile
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == ["00000001.txt"]
        assert archive.read("00000001.txt") == b""
    detection = torch.tensor([[16., 16., 48., 48., 0.8, 2.]])
    rows = detection_lines(detection, (64, 64))
    assert rows == ["2 0.50000000 0.50000000 0.50000000 0.50000000 0.80000001"]


def test_prediction_auto_rect_matches_ultralytics_letterbox_and_box_scaling(tmp_path):
    record = write_record(tmp_path, ".png")
    for path in record[1:4]:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert cv2.imwrite(str(path), cv2.resize(raw, (96, 64)))
    image, original_shape, ratio_pad = prepare_image(
        *record[1:4], imgsz=128, letterbox_mode="auto_rect"
    )
    rgb = cv2.imread(str(record[1]), cv2.IMREAD_COLOR)
    expected = LetterBox((128, 128), auto=True, stride=32, scaleup=True)(image=rgb)
    assert image.shape[1:] == expected.shape[:2]
    assert torch.equal(torch.from_numpy(image[:3]),
                       torch.from_numpy(expected[..., ::-1].transpose(2, 0, 1).copy()))
    from ultralytics.utils.ops import scale_boxes
    gain, padding = ratio_pad
    boxes = torch.tensor([[12. * gain[0] + padding[0],
                           12. * gain[1] + padding[1],
                           72. * gain[0] + padding[0],
                           48. * gain[1] + padding[1]]])
    restored = scale_boxes(image.shape[1:], boxes.clone(), original_shape,
                           ratio_pad=ratio_pad)
    assert torch.allclose(restored, torch.tensor([[12., 12., 72., 48.]]), atol=1e-4)
    square, _, _ = prepare_image(*record[1:4], imgsz=128, letterbox_mode="square")
    assert square.shape == (11, 128, 128)


def test_prediction_device_zero_means_first_cuda_device_and_cpu_stays_cpu():
    assert resolve_device("cpu") == torch.device("cpu")
    if torch.cuda.is_available():
        assert resolve_device("0") == torch.device("cuda:0")
    else:
        import pytest
        with pytest.raises(RuntimeError, match="CUDA was requested"):
            resolve_device("0")


def test_rectangular_validation_groups_aspect_ratios_and_keeps_ratio_metadata(tmp_path):
    records = []
    for index, shape in enumerate(((64, 128), (128, 64))):
        directory = tmp_path / str(index)
        directory.mkdir()
        record = write_record(directory, ".png")
        for path in record[1:4]:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            resized = cv2.resize(image, shape[::-1], interpolation=cv2.INTER_NEAREST)
            assert cv2.imwrite(str(path), resized)
        records.append(record)
    dataset = QualityTriModalDataset(records, 128, make_hyp(), augment=False,
                                    rect_batch_size=1)
    assert dataset.rect
    first, second = dataset[0], dataset[1]
    assert first["img"].shape[1:] == (96, 160)
    assert second["img"].shape[1:] == (160, 96)
    assert first["ratio_pad"][0] == (1.0, 1.0)
    assert second["ratio_pad"][0] == (1.0, 1.0)


def test_audit_ablation_changes_only_the_requested_modality():
    assert M960_RECT_VAL_BATCH == 16
    class Validator:
        def preprocess(self, batch):
            return batch

        def __call__(self, trainer):
            self.observed = self.preprocess({"img": torch.ones(1, 11, 32, 32)})["img"]
            return {"metrics/mAP50(B)": 0.5, "metrics/mAP50-95(B)": 0.3,
                    "metrics/precision(B)": 0.6, "metrics/recall(B)": 0.4}

    class Trainer:
        def get_validator(self):
            self.validator = Validator()
            return self.validator

    trainer = Trainer()
    loader = SimpleNamespace(dataset=[None])
    validate_mode(trainer, loader, "IR_OFF")
    assert trainer.validator.observed[:, 3:6].count_nonzero() == 0
    assert trainer.validator.observed[:, 9].count_nonzero() == 0
    assert trainer.validator.observed[:, 6:9].min() == 1
    validate_mode(trainer, loader, "DEPTH_OFF")
    assert trainer.validator.observed[:, 6:9].count_nonzero() == 0
    assert trainer.validator.observed[:, 3:6].min() == 1
    assert trainer.validator.observed[:, 9].min() == 1


def test_rgb_protocol_resize_matches_ultralytics_val_rgb_pixels(tmp_path):
    record = write_record(tmp_path, ".jpg")
    for path in record[1:4]:
        original = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert cv2.imwrite(str(path), cv2.resize(original, (96, 64)))
    image_dir = tmp_path / "yolo" / "images" / "val"
    label_dir = tmp_path / "yolo" / "labels" / "val"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    (image_dir / record[1].name).write_bytes(record[1].read_bytes())
    (label_dir / (record[1].stem + ".txt")).write_bytes(record[4].read_bytes())
    reference = YOLODataset(
        img_path=str(image_dir), imgsz=128, cache=False, augment=False,
        hyp=get_cfg(), rect=False, batch_size=1, stride=32,
        data={"channels": 3, "names": {0: "person"}},
    )[0]
    legacy = QualityTriModalDataset([record], 128, make_hyp(), augment=False)[0]
    corrected = QualityTriModalDataset([record], 128, make_hyp(), augment=False,
                                       rgb_protocol_resize=True)[0]
    assert not torch.equal(legacy["img"][:3], reference["img"])
    assert torch.equal(corrected["img"][:3], reference["img"])
    assert corrected["ratio_pad"] == reference["ratio_pad"]
    reference_rect = YOLODataset(
        img_path=str(image_dir), imgsz=128, cache=False, augment=False,
        hyp=get_cfg(), rect=True, batch_size=1, stride=32,
        data={"channels": 3, "names": {0: "person"}},
    )[0]
    corrected_rect = QualityTriModalDataset(
        [record], 128, make_hyp(), augment=False, rect_batch_size=1,
        rgb_protocol_resize=True,
    )[0]
    assert torch.equal(corrected_rect["img"][:3], reference_rect["img"])
    assert corrected_rect["ratio_pad"] == reference_rect["ratio_pad"]
