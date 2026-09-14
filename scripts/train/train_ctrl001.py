"""Safe CTRL001 entry: canonical check-only by default; training requires reviewed SHA."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import ultralytics
import yaml
from ultralytics.cfg import get_cfg
from ultralytics.utils.torch_utils import init_seeds


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.fusion.control import RGBControlTrainer, initialize_rgb_control  # noqa: E402
from src.fusion.diagnostics import compare_ctrl001_to_f001  # noqa: E402
from src.fusion.initialization import INITIAL_PATH, verify_checkpoint  # noqa: E402
from src.fusion.runtime import git_state, require_reviewed_checkout  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/experiments/CTRL001_RGB_F001_AUG.yaml"
F001_CONFIG = ROOT / "configs/experiments/F001_RGB_IR_GATED_P45_YOLO11N.yaml"


def project_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_reviewed_config(path: Path) -> Dict[str, Any]:
    comparison = compare_ctrl001_to_f001(path, F001_CONFIG)
    if not comparison["passed"]:
        raise RuntimeError("CTRL001/F001 machine comparison failed: " + json.dumps(comparison, sort_keys=True))
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Validate real data and initialization (default)")
    mode.add_argument("--train", action="store_true", help="Enter the reviewed CTRL001 DetectionTrainer")
    parser.add_argument("--expected-sha", help="Required exact reviewed commit for --train")
    args = parser.parse_args(argv)
    if args.train and not args.expected_sha:
        parser.error("--expected-sha is required with --train")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_reviewed_config(args.config.resolve())
    train_config = dict(config["train"])
    commit, dirty, branch = git_state(ROOT)
    if args.train:
        require_reviewed_checkout(args.expected_sha, (commit, dirty, branch))

    checkpoint_sha = verify_checkpoint(INITIAL_PATH)
    provenance = {
        "experiment_id": config["experiment_id"],
        "mode": "train" if args.train else "check-only",
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": dirty,
        "expected_sha": args.expected_sha,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "model_configuration": config["architecture"],
        "data_configuration": config["data_configuration"],
        "initial_checkpoint_sha256": checkpoint_sha,
    }
    print(json.dumps(provenance, indent=2, sort_keys=True))

    train_config["pretrained"] = str(INITIAL_PATH)
    train_config["data"] = str(project_path(train_config["data"]))
    train_config["project"] = str(project_path(train_config["project"]))
    if args.train:
        trainer = RGBControlTrainer(overrides=train_config)
        trainer.train()
        return 0

    # Avoid BaseTrainer side effects and prove that the real canonical adapter/model initialize.
    trainer = RGBControlTrainer.__new__(RGBControlTrainer)
    trainer.args = get_cfg(overrides=train_config)
    trainer.args.device = "cpu"
    trainer.args.workers = 0
    trainer.args.batch = 1
    trainer.data = trainer.get_dataset()
    init_seeds(2026, deterministic=True)
    trainer.model = trainer.get_model(verbose=False)
    initialize_rgb_control(trainer.model, INITIAL_PATH, verbose=False)
    print("CHECK-ONLY PASS: canonical records, paired RGB transform adapter, graph and initialization verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
