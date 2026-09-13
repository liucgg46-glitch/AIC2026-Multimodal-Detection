from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from ultralytics.cfg import get_cfg
from ultralytics.nn import tasks as nn_tasks
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import callbacks as callback_module
from ultralytics.utils.torch_utils import ModelEMA, strip_optimizer

from src.fusion.dual_stream_model import DualStreamModel
from src.fusion.paired_dataset import PairedDataset
from src.fusion.runtime import (
    formal_iterations,
    one_image_loader,
    require_reviewed_checkout,
    resolved_runtime_provenance,
)
from src.fusion.trainer import FusionTrainer


ROOT = Path(__file__).resolve().parents[2]
HEAD = "6fb1a3d50c216797d49a0a3a7bacf4c4c6ce4597"


class TinyDataset(Dataset):
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return torch.tensor(index)

    @staticmethod
    def collate_fn(items):
        return items


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1)
        self.stride = torch.tensor([32.0])
        self.end2end = False

    def forward(self, value):
        return self.conv(value)

    def set_head_attr(self, **kwargs):
        pass


class DummyMetrics:
    keys = []


class DummyValidator:
    metrics = DummyMetrics()


def test_server_smoke_uses_real_setup_optimizer_and_checkpoint_apis():
    source = (ROOT / "scripts/train/smoke_fusion_server.py").read_text(encoding="utf-8")
    assert "FusionTrainer(overrides=" in source
    assert "trainer._setup_train()" in source
    assert "trainer.optimizer_step()" in source
    assert "trainer.save_model()" in source
    assert "strip_optimizer(" in source
    assert "load_checkpoint(" in source
    assert "require_optimizer_state=True" in source
    # One definition plus exactly two calls from main.
    assert source.count("_optimization_cycle(trainer, train_iter") == 3
    assert '"ema_updates_delta"' in source
    assert '"two_training_steps_passed": True' in source
    assert "torch.optim.SGD" not in source
    assert "FusionTrainer.__new__" not in source
    assert "trainer.train()" not in source


@pytest.mark.parametrize(
    "state,message",
    [
        ((HEAD, False, "feature/multimodal-fusion"), "detached HEAD"),
        ((("0" * 40), False, "DETACHED"), "does not match"),
        ((HEAD, True, "DETACHED"), "clean working tree"),
    ],
)
def test_reviewed_checkout_rejects_branch_wrong_sha_and_dirty_detached(state, message):
    with pytest.raises(RuntimeError, match=message):
        require_reviewed_checkout(HEAD, state)


def test_reviewed_checkout_accepts_only_clean_exact_detached():
    assert require_reviewed_checkout(HEAD.upper(), (HEAD, False, "DETACHED")) == HEAD
    with pytest.raises(RuntimeError, match="40-character"):
        require_reviewed_checkout("main", (HEAD, False, "DETACHED"))


def test_real_upstream_setup_resolves_adamw_accumulation_and_val_batch(monkeypatch, tmp_path):
    monkeypatch.setattr(callback_module, "add_integration_callbacks", lambda instance: None)
    monkeypatch.setattr(
        FusionTrainer,
        "get_dataset",
        lambda self: {"train": "train", "val": "val", "nc": 12, "channels": 6,
                      "names": dict(enumerate(str(i) for i in range(12)))},
    )

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        length = 1600 if mode == "train" else 400
        return DataLoader(TinyDataset(length), batch_size=batch_size, shuffle=False)

    monkeypatch.setattr(FusionTrainer, "get_dataloader", get_dataloader)
    monkeypatch.setattr(FusionTrainer, "get_validator", lambda self: DummyValidator())
    callbacks = {"on_pretrain_routine_start": [], "on_pretrain_routine_end": []}
    trainer = FusionTrainer(
        overrides={
            "model": "yolo11n.yaml", "data": "synthetic.yaml", "device": "cpu", "amp": False,
            "pretrained": False, "batch": 32, "nbs": 64, "epochs": 100, "optimizer": "auto",
            "workers": 0, "plots": False, "compile": False, "project": str(tmp_path),
            "name": "setup", "exist_ok": True,
        },
        _callbacks=callbacks,
    )
    trainer.model = TinyModel()
    trainer.callbacks = callbacks
    trainer._setup_train()

    report = resolved_runtime_provenance(trainer, "auto")
    assert formal_iterations(1600, 32, 64, 100) == 2500
    assert report["formal_iterations"] == 2500
    assert report["requested_optimizer"] == "auto"
    assert report["resolved_optimizer"] == "AdamW"
    assert report["physical_train_batch"] == 32
    assert report["resolved_val_batch"] == 64
    assert report["nbs"] == 64
    assert report["accumulate"] == 2
    assert report["nominal_effective_batch"] == 64
    assert report["ema_enabled"] is True
    assert report["scheduler"] == "LambdaLR"
    assert len(report["optimizer_parameter_groups"]) == 3


def test_one_image_loader_is_really_length_one():
    loader = one_image_loader(TinyDataset(3))
    assert len(loader.dataset) == 1
    assert next(iter(loader)) == [torch.tensor(0)]


def _disabled_scaler():
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=False)
    return torch.cuda.amp.GradScaler(enabled=False)


class TinyLossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        loss = ((self.weight * batch["x"]) - batch["target"]).square().sum()
        return loss, loss.detach().reshape(1)


def test_second_optimization_cycle_starts_with_resident_adamw_state():
    from scripts.train.smoke_fusion_server import _optimization_cycle

    trainer = FusionTrainer.__new__(FusionTrainer)
    trainer.model = TinyLossModel()
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)
    trainer.scaler = _disabled_scaler()
    trainer.ema = ModelEMA(trainer.model)
    trainer.amp = False
    trainer.device = torch.device("cpu")
    trainer.preprocess_batch = lambda batch: batch
    train_iter = iter([
        {"x": torch.tensor([2.0]), "target": torch.tensor([0.0])},
        {"x": torch.tensor([3.0]), "target": torch.tensor([0.0])},
    ])

    ema_before = trainer.ema.updates
    step1 = _optimization_cycle(trainer, train_iter)
    assert step1["optimizer_state_entries_before"] == 0
    assert step1["optimizer_state_entries_after"] > 0
    step2 = _optimization_cycle(trainer, train_iter, require_optimizer_state=True)
    assert step2["optimizer_state_entries_before"] == step1["optimizer_state_entries_after"]
    assert step2["optimizer_state_entries_after"] >= step1["optimizer_state_entries_after"]
    assert trainer.ema.updates - ema_before == 2
    assert np.isfinite(step2["loss"])


def test_ultralytics_checkpoint_path_restores_and_validates_one_image(tmp_path, hyp, monkeypatch):
    image = np.full((32, 48, 3), 100, np.uint8)
    rgb, ir, label = tmp_path / "rgb.png", tmp_path / "ir.png", tmp_path / "label.txt"
    cv2.imwrite(str(rgb), image)
    cv2.imwrite(str(ir), image)
    label.write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")
    dataset = PairedDataset([("one", rgb, ir, label)], 64, hyp)
    loader = one_image_loader(dataset)

    args = get_cfg(overrides={
        "model": "yolo11n.yaml", "data": str(ROOT / "configs/data/F001_RGB_IR_RAW3.yaml"),
        "task": "detect", "device": "cpu", "imgsz": 64, "batch": 1, "workers": 0,
        "half": False, "amp": False, "plots": False, "save_json": False, "compile": False,
    })
    model = DualStreamModel().eval()
    model.args = args
    trainer = FusionTrainer.__new__(FusionTrainer)
    trainer.args = args
    trainer.model = model
    trainer.ema = ModelEMA(model)
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer.scaler = _disabled_scaler()
    trainer.metrics = {}
    trainer.fitness = trainer.best_fitness = 0.0
    trainer.epoch = 0
    trainer.save_period = -1
    trainer.wdir = tmp_path / "weights"
    trainer.last, trainer.best = trainer.wdir / "last.pt", trainer.wdir / "best.pt"
    trainer.csv = tmp_path / "results.csv"
    trainer.save_model()
    stripped = strip_optimizer(str(trainer.best))
    assert stripped["epoch"] == -1
    assert stripped["model"] is not None
    assert stripped["optimizer"] is None
    assert stripped["ema"] is None
    assert stripped["scaler"] is None

    restored, checkpoint = load_checkpoint(str(trainer.best), device=torch.device("cpu"), fuse=False)
    assert isinstance(restored, DualStreamModel)
    assert checkpoint["ema"] is None and checkpoint["model"] is not None
    with torch.inference_mode():
        assert restored(torch.zeros(1, 6, 64, 64))[0].shape == (1, 16, 84)

    calls = []
    real_load_checkpoint = nn_tasks.load_checkpoint

    def tracked_load_checkpoint(weight, *args, **kwargs):
        calls.append(str(weight))
        return real_load_checkpoint(weight, *args, **kwargs)

    monkeypatch.setattr(nn_tasks, "load_checkpoint", tracked_load_checkpoint)
    validator = DetectionValidator(loader, save_dir=tmp_path / "validator", args=deepcopy(args))
    metrics = validator(model=str(trainer.best))
    assert len(validator.dataloader.dataset) == 1
    assert calls == [str(trainer.best)]
    assert "metrics/mAP50-95(B)" in metrics
