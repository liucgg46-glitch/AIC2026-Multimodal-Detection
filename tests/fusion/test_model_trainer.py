from copy import deepcopy
import io
from types import SimpleNamespace

import numpy as np
import cv2
import pytest
import torch
from torch.utils.data import DataLoader
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator

from src.fusion.dual_stream_model import DualStreamModel
from src.fusion.initialization import INITIAL_PATH, INITIAL_SHA256, initialize_canonical
from src.fusion.paired_dataset import PairedDataset
from src.fusion.trainer import FusionTrainer


@pytest.fixture
def model(hyp):
    return DualStreamModel().eval()


def test_split_and_independent_encoders(model):
    seen = {}
    def record(name):
        def hook(module, args):
            seen[name] = args[0].clone()
        return hook
    hooks = [model.model[0].register_forward_pre_hook(record("rgb")),
             model.ir_encoder[0].register_forward_pre_hook(record("ir"))]
    x = torch.cat([torch.ones(1, 3, 64, 64), torch.full((1, 3, 64, 64), 2.)], 1)
    with torch.inference_mode():
        model(x)
    for hook in hooks:
        hook.remove()
    assert torch.equal(seen["rgb"], x[:, :3])
    assert torch.equal(seen["ir"], x[:, 3:])
    assert model.model[0].conv.weight.data_ptr() != model.ir_encoder[0].conv.weight.data_ptr()
    with pytest.raises(ValueError, match="6"):
        model(x[:, :3])


def test_fusion_shapes_and_zero_fallback(model):
    x = torch.rand(2, 6, 128, 128)
    with torch.inference_mode():
        rgb, ir = model.backbone_features(x)
        fused = model.fused_features(x)
        for index, shape in ((4, (2, 128, 16, 16)), (6, (2, 128, 8, 8)), (10, (2, 256, 4, 4))):
            assert tuple(fused[index].shape) == shape
            torch.testing.assert_close(fused[index], rgb[index], rtol=0, atol=0)
        for module, index in ((model.fusion4, 6), (model.fusion5, 10)):
            assert torch.count_nonzero(module.proj(ir[index])) == 0
            gates = module.gate(torch.cat((rgb[index], ir[index]), 1))
            assert gates.min() >= 0 and gates.max() <= 1
        # A separate 12-class RGB model with the exact same RGB graph weights is the reference.
        reference = DetectionModel("yolo11n.yaml", nc=12, verbose=False).eval()
        reference.model.load_state_dict(model.model.state_dict(), strict=True)
        expected = reference(x[:, :3])[0]
        actual = model(x)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.shape == (2, 16, 336)


def test_p3_rgb_only_and_nonzero_ir_path(model):
    with torch.no_grad():
        model.fusion4.proj.weight.fill_(0.01)
        model.fusion5.proj.weight.fill_(0.01)
        model.fusion4.proj.bias.fill_(0.1)
    x = torch.rand(1, 6, 64, 64)
    with torch.inference_mode():
        rgb, ir = model.backbone_features(x)
        f = model.fused_features(x)
        torch.testing.assert_close(f[4], rgb[4], rtol=0, atol=0)
        assert not torch.equal(f[6], rgb[6])
        torch.testing.assert_close(f[6], rgb[6] + model.fusion4.gate(torch.cat((rgb[6], ir[6]), 1)) * model.fusion4.proj(ir[6]))


def test_canonical_pretrained_load_report_and_fallback(model):
    source = torch.load(str(INITIAL_PATH), map_location="cpu", weights_only=False)["model"].float()
    report = initialize_canonical(model, verbose=False)
    assert report["checkpoint_sha256"] == INITIAL_SHA256
    assert report["unexpected_keys"] == []
    assert report["ratios"]["rgb_encoder"]["key_ratio"] == 1.0
    assert report["ratios"]["ir_encoder"]["key_ratio"] == 1.0
    assert 0.79 < report["ratios"]["neck_head"]["key_ratio"] < 0.82
    assert report["shape_mismatch_keys"]
    assert all(row["key"].startswith("model.23.cv3.") for row in report["shape_mismatch_keys"])
    for index in (0, 4, 10):
        key = "model.%d" % index
        rgb_state = model.model[index].state_dict()
        ir_state = model.ir_encoder[index].state_dict()
        source_state = source.model[index].state_dict()
        for name in rgb_state:
            torch.testing.assert_close(rgb_state[name], source_state[name])
            torch.testing.assert_close(ir_state[name], source_state[name])
    for fusion in (model.fusion4, model.fusion5):
        assert torch.count_nonzero(fusion.proj.weight) == 0
        assert torch.count_nonzero(fusion.proj.bias) == 0
        assert torch.count_nonzero(fusion.gate[0].weight) == 0
        assert torch.count_nonzero(fusion.gate[0].bias) == 0
        assert fusion.gate(torch.zeros(1, fusion.proj.in_channels * 2, 2, 2)).unique().item() == 0.5
    x = torch.rand(1, 6, 64, 64)
    reference = DetectionModel("yolo11n.yaml", nc=12, verbose=False).eval()
    reference.model.load_state_dict(model.model.state_dict(), strict=True)
    model.eval()
    with torch.inference_mode():
        # This is equality to the same initialized 12-class RGB graph, not the canonical 80-class checkpoint output.
        torch.testing.assert_close(model(x)[0], reference(x[:, :3])[0], rtol=0, atol=0)


def test_weight_serialization(model):
    with torch.no_grad():
        model.fusion4.proj.bias.fill_(0.3)
    restored = DualStreamModel()
    restored.load(model, verbose=False)
    torch.testing.assert_close(restored.fusion4.proj.bias, model.fusion4.proj.bias)
    stream = io.BytesIO()
    torch.save({"model": model}, stream)
    stream.seek(0)
    loaded = torch.load(stream, weights_only=False)["model"]
    with torch.inference_mode():
        x = torch.rand(1, 6, 64, 64)
        torch.testing.assert_close(loaded(x)[0], model(x)[0])


def test_cpu_pipeline_loss_and_validator(tmp_path, hyp, model):
    image = np.full((32, 48, 3), 100, np.uint8)
    rgb, ir, label = tmp_path / "rgb.png", tmp_path / "ir.png", tmp_path / "label.txt"
    for path in (rgb, ir):
        cv2.imwrite(str(path), image)
    label.write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")
    dataset = PairedDataset([("one", rgb, ir, label)], 64, hyp)
    batch = next(iter(DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)))
    trainer = FusionTrainer.__new__(FusionTrainer)
    trainer.args, trainer.device = hyp, torch.device("cpu")
    validator = DetectionValidator.__new__(DetectionValidator)
    validator.device, validator.args = trainer.device, SimpleNamespace(quantize=32, half=False)
    validated = validator.preprocess(deepcopy(batch))
    batch = trainer.preprocess_batch(batch)
    assert batch["img"].shape == (1, 6, 64, 64)
    torch.testing.assert_close(validated["img"], batch["img"])
    model.args = hyp
    with torch.inference_mode():
        preds = model(batch["img"])
        loss, items = model.loss(batch, preds)
    assert torch.isfinite(preds[0]).all() and torch.isfinite(loss).all()
    if isinstance(items, dict):
        assert all(torch.isfinite(value).all() for value in items.values())
    else:
        assert torch.isfinite(items).all()  # 8.3.253 returns a three-element tensor
    # The optimization loop, validation metrics and criterion remain upstream implementations.
    assert FusionTrainer.train is DetectionTrainer.train
    assert DualStreamModel.init_criterion is DetectionModel.init_criterion


def test_pyramid_semantics_and_detect_inputs(model):
    captured = {}
    def capture(module, args):
        captured["features"] = args[0]
    handle = model.model[23].register_forward_pre_hook(capture)
    x = torch.rand(1, 6, 160, 160)
    with torch.inference_mode():
        backbone = model.fused_features(x)
        model(x)
    handle.remove()
    assert {i: tuple(backbone[i].shape[-2:]) for i in (4, 6, 10)} == {4: (20, 20), 6: (10, 10), 10: (5, 5)}
    assert [tuple(feature.shape[-2:]) for feature in captured["features"]] == [(20, 20), (10, 10), (5, 5)]
    assert model.model[12].f == [-1, 6]  # neck consumes P4
    assert model.model[15].f == [-1, 4]  # neck consumes P3
    assert model.model[21].f == [-1, 10]  # neck consumes P5
    assert model.model[23].f == [16, 19, 22]
    assert model.stride.tolist() == [8.0, 16.0, 32.0]


def test_upstream_final_validator_on_paired_loader(tmp_path, hyp, model):
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    image = np.full((32, 48, 3), 100, np.uint8)
    path, label = tmp_path / "one.png", tmp_path / "one.txt"
    cv2.imwrite(str(path), image)
    label.write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")
    dataset = PairedDataset([("one", path, path, label)], 64, hyp)
    loader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)
    hyp.data = str(root / "configs/data/F001_RGB_IR_RAW3.yaml")
    hyp.model, hyp.imgsz, hyp.device, hyp.plots = "yolo11n.yaml", 64, "cpu", False
    validator = DetectionValidator(loader, save_dir=tmp_path, args=hyp)
    results = validator(model=model)
    assert "metrics/mAP50-95(B)" in results
