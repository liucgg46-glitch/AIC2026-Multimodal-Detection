"""Safe F001 entry: check-only by default, explicit smoke or formal training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torchvision
import ultralytics
import yaml
from ultralytics.cfg import get_cfg
from ultralytics.utils.torch_utils import init_seeds
from src.fusion.trainer import FusionTrainer
from src.fusion.initialization import INITIAL_PATH, INITIAL_SHA256, initialize_canonical, verify_checkpoint
from src.fusion.runtime import git_state, print_resolved_runtime, require_reviewed_checkout


def load_config(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected_architecture = dict(encoders="independent_yolo11n", fusion_levels=["P4", "P5"],
                                 p3="rgb_backbone_only", zero_init=True,
                                 pretrained="weights/yolo11n.pt", pretrained_sha256=INITIAL_SHA256)
    expected_data = dict(rgb="data/raw/train/visible", ir="data/raw/train/infrared",
                         ir_representation="E002_raw3_byte_preserving_source",
                         labels="data/processed/train/labels_clean", train="data/splits/train.txt",
                         val="data/splits/val.txt", counts=[1600, 400])
    if config["architecture"] != expected_architecture or config["data_configuration"] != expected_data:
        raise ValueError("F001 v1 model/data configuration differs from the implemented contract")
    if config["train"]["pretrained"] != "weights/yolo11n.pt":
        raise ValueError("F001 requires canonical weights/yolo11n.pt")
    formal = {"model": "yolo11n.yaml", "seed": 2026, "epochs": 100, "batch": 32,
              "imgsz": 640, "workers": 8, "optimizer": "auto", "nbs": 64}
    if any(config["train"].get(key) != value for key, value in formal.items()):
        raise ValueError("F001 formal training contract was changed")
    batch_policy = config.get("batch_policy", {})
    if (batch_policy.get("canonical_target_batch") != 32 or
            batch_policy.get("server_feasibility_sequence") != [1, 2, 8, 16, 32] or
            batch_policy.get("formal_accumulate_after_warmup") != 2):
        raise ValueError("F001 batch policy was changed")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/F001_RGB_IR_GATED_P45_YOLO11N.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Validate provenance, checkpoint and pairing (default)")
    mode.add_argument("--smoke", action="store_true", help="One local CPU forward; never calls train()")
    mode.add_argument("--train", action="store_true", help="Enter the real Ultralytics training pipeline")
    parser.add_argument("--expected-sha", help="Required exact reviewed commit for --train")
    parser.add_argument("--imgsz", type=int, default=64, help="Smoke resolution only")
    args = parser.parse_args()
    if args.train and not args.expected_sha:
        parser.error("--expected-sha is required with --train")
    config = load_config(ROOT / args.config)
    commit, dirty, branch = git_state(ROOT)
    if args.train:
        require_reviewed_checkout(args.expected_sha, (commit, dirty, branch))
    elif branch not in ("feature/multimodal-fusion", "DETACHED"):
        raise ValueError("F001 must run from feature/multimodal-fusion or a detached reviewed commit")
    checkpoint_sha = verify_checkpoint(INITIAL_PATH)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    provenance = dict(experiment_id=config["experiment_id"], git_commit=commit, git_branch=branch, git_dirty=dirty,
                      expected_sha=args.expected_sha,
                      python=sys.version.split()[0], torch=torch.__version__, torchvision=torchvision.__version__,
                      ultralytics=ultralytics.__version__, cuda_available=torch.cuda.is_available(), gpu=gpu,
                      model_configuration=config["architecture"], data_configuration=config["data_configuration"],
                      initial_checkpoint="weights/yolo11n.pt", initial_checkpoint_sha256=checkpoint_sha,
                      augmentation_policy={k: config["train"][k] for k in
                                           ("degrees", "translate", "scale", "shear", "perspective", "mosaic",
                                            "mixup", "cutmix", "copy_paste", "multi_scale", "bgr", "fliplr",
                                            "flipud", "hsv_h", "hsv_s", "hsv_v")},
                      seed=config["train"]["seed"], epochs=config["train"]["epochs"],
                      batch=config["train"]["batch"], imgsz=config["train"]["imgsz"],
                      workers=config["train"]["workers"], optimizer=config["train"]["optimizer"],
                      nbs=config["train"]["nbs"],
                      batch_policy=config["batch_policy"], mode="train" if args.train else "smoke" if args.smoke else "check-only")
    print(json.dumps(provenance, indent=2))
    if args.train:
        train_overrides = dict(config["train"])
        train_overrides["pretrained"] = str(INITIAL_PATH)
        train_overrides["project"] = str((ROOT / config["train"]["project"]).resolve())
        trainer = FusionTrainer(overrides=train_overrides)
        requested_optimizer = train_overrides["optimizer"]
        trainer.add_callback(
            "on_pretrain_routine_end",
            lambda active_trainer: print_resolved_runtime(active_trainer, requested_optimizer),
        )
        trainer.train()
        return 0
    # No BaseTrainer constructor: avoid run directories, integrations or training setup for forward smoke.
    trainer = FusionTrainer.__new__(FusionTrainer)
    trainer.args = get_cfg(overrides=config["train"])
    if not args.train:
        trainer.args.device = "cpu"
        trainer.args.amp = False
        trainer.args.workers = 0
        trainer.args.batch = 1
    init_seeds(2026, deterministic=True)
    trainer.data = trainer.get_dataset()
    print(f"Pairing: train={len(trainer.paired_records['train'])}, val={len(trainer.paired_records['val'])}")
    if args.smoke:
        torch.set_num_threads(2)
        trainer.args.imgsz = args.imgsz
        trainer.device = torch.device("cpu")
        trainer.model = trainer.get_model(verbose=False)
        initialize_canonical(trainer.model, verbose=True)
        trainer.model.eval()
        loader = trainer.get_dataloader(trainer.data["val"], batch_size=1, rank=-1, mode="val")
        batch = trainer.preprocess_batch(next(iter(loader)))
        with torch.inference_mode():
            output = trainer.model(batch["img"])
        print(f"CPU forward PASS: batch={tuple(batch['img'].shape)}, predictions={tuple(output[0].shape)}")
    else:
        trainer.model = trainer.get_model(verbose=False)
        initialize_canonical(trainer.model, verbose=True)
        print("CHECK-ONLY PASS: checkpoint, graph, initialization and pairing verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
