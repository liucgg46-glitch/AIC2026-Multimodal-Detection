"""Real-data audit: requires read access to the local official 2000 paired images."""
from pathlib import Path

import pytest
import torch
from ultralytics.cfg import get_cfg

from src.fusion.dual_stream_model import DualStreamModel
from src.fusion.initialization import INITIAL_PATH, INITIAL_SHA256
from src.fusion.trainer import FusionTrainer
from scripts.train.train_fusion import load_config

ROOT = Path(__file__).resolve().parents[2]


def make_trainer():
    config = load_config(ROOT / "configs/experiments/F001_RGB_IR_GATED_P45_YOLO11N.yaml")
    trainer = FusionTrainer.__new__(FusionTrainer)
    trainer.args = get_cfg(overrides=config["train"])
    trainer.args.workers = 0
    trainer.device = torch.device("cpu")
    return trainer


def test_real_2000_decode_and_forward():
    torch.set_num_threads(2)
    trainer = make_trainer()
    trainer.args.imgsz = 64
    trainer.data = trainer.get_dataset()
    trainer.model = trainer.get_model(verbose=False).eval()
    for mode, count in (("train", 1600), ("val", 400)):
        # Constructor decodes EVERY RGB and IR image and checks shapes/encoding/labels.
        loader = trainer.get_dataloader(trainer.data[mode], batch_size=1, rank=-1, mode=mode)
        assert len(loader.dataset) == count
        batch = trainer.preprocess_batch(next(iter(loader)))
        assert batch["img"].shape == (1, 6, 64, 64)
        with torch.inference_mode():
            preds = trainer.model(batch["img"])
        assert preds[0].shape == (1, 16, 84)
        assert torch.isfinite(preds[0]).all()


@pytest.mark.parametrize("key,value", [("seed", 7), ("mosaic", 1.0), ("plots", True),
                                      ("resume", True), ("fraction", 0.5),
                                      ("model", "yolo11s.yaml"), ("close_mosaic", 10)])
def test_trainer_rejects_unsupported_options(key, value):
    trainer = make_trainer()
    setattr(trainer.args, key, value)
    with pytest.raises(ValueError):
        trainer.get_dataset()


def test_real_base_trainer_setup_uses_canonical_initialization(tmp_path):
    config = load_config(ROOT / "configs/experiments/F001_RGB_IR_GATED_P45_YOLO11N.yaml")
    overrides = dict(config["train"])
    overrides.update(project=str(tmp_path), name="setup", exist_ok=True, device="cpu", workers=0,
                     batch=1, amp=False, epochs=1, save=False, verbose=False,
                     pretrained=str(INITIAL_PATH))
    trainer = FusionTrainer(overrides=overrides)
    trainer.setup_model()
    assert isinstance(trainer.model, DualStreamModel)
    assert trainer.model.load_report["checkpoint_sha256"] == INITIAL_SHA256
