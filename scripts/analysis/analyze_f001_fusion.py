"""Run read-only F001 inference ablations and full-val fusion-state statistics."""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.fusion.diagnostics import (  # noqa: E402
    ABLATION_MODES,
    F001_CHECKPOINT_SHA256,
    F001_TRAINING_SHA,
    AblationDataset,
    collect_fusion_statistics,
    deterministic_ir_shuffle,
    fixed_val_records,
    immutable_checkpoint,
    load_f001_model,
    require_new_output_dir,
    run_ablation_mode,
    run_fresh_model_plan,
    sha256,
    validation_args,
    write_csv,
    write_json,
)
from src.fusion.paired_dataset import PairedDataset  # noqa: E402
from src.fusion.runtime import git_state, require_reviewed_checkout  # noqa: E402


DEFAULT_CHECKPOINT = ROOT / "runs/F001_RGB_IR_GATED_P45_YOLO11N/weights/best.pt"
DEFAULT_OUTPUT = ROOT / "outputs/analysis/F001_ablation"
F001_CONFIG = ROOT / "configs/experiments/F001_RGB_IR_GATED_P45_YOLO11N.yaml"


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--expected-checkpoint-sha256", default=F001_CHECKPOINT_SHA256)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--expected-sha", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> Dict[str, Any]:
    current_sha = require_reviewed_checkout(args.expected_sha, git_state(ROOT))
    checkpoint = args.checkpoint.resolve()
    config = yaml.safe_load(F001_CONFIG.read_text(encoding="utf-8"))
    records = fixed_val_records()
    shuffled_records, shuffle_manifest = deterministic_ir_shuffle(records, args.seed)

    with immutable_checkpoint(checkpoint, args.expected_checkpoint_sha256) as checkpoint_hash:
        val_args = validation_args(config, args.device, args.batch, args.workers)
        base = PairedDataset(records, val_args.imgsz, val_args, augment=False)
        shuffled = PairedDataset(shuffled_records, val_args.imgsz, val_args, augment=False)
        if [row[0] for row in base.records] != [row[0] for row in shuffled.records]:
            raise RuntimeError("IR shuffle changed fixed val RGB stem order")
        output_dir = require_new_output_dir(args.output_dir)
        write_csv(
            output_dir / "ir_shuffle_manifest.csv",
            shuffle_manifest,
            ("rgb_stem", "original_ir_stem", "shuffled_ir_stem", "height", "width"),
        )

        def run_mode(model, mode):
            mode_base = shuffled if mode == "IR_SHUFFLED" else base
            dataset = AblationDataset(mode_base, mode)
            before = sha256(checkpoint)
            report = run_ablation_mode(
                model,
                dataset,
                mode,
                deepcopy(val_args),
                output_dir / ("validator_" + mode.lower()),
                checkpoint_hash,
                current_sha,
                args.seed,
            )
            after = sha256(checkpoint)
            if before != after or after != checkpoint_hash:
                raise RuntimeError("Checkpoint changed during %s ablation" % mode)
            report["checkpoint_unchanged"] = True
            write_json(output_dir / (mode.lower() + ".json"), report)
            return report

        def run_statistics(model):
            return collect_fusion_statistics(
                model, AblationDataset(base, "NORMAL"), deepcopy(val_args), checkpoint_hash, current_sha
            )

        mode_reports, statistics = run_fresh_model_plan(
            ABLATION_MODES,
            lambda: load_f001_model(checkpoint, args.device),
            run_mode,
            run_statistics,
        )
        write_json(output_dir / "fusion_statistics.json", statistics)

        comparison_rows = [
            {
                "mode": report["mode"],
                "precision": report["precision"],
                "recall": report["recall"],
                "mAP50": report["map50"],
                "mAP50-95": report["map50_95"],
                "sample_count": report["sample_count"],
                "gt_instance_count": report["gt_instance_count"],
            }
            for report in mode_reports
        ]
        write_csv(
            output_dir / "comparison.csv",
            comparison_rows,
            ("mode", "precision", "recall", "mAP50", "mAP50-95", "sample_count", "gt_instance_count"),
        )
        provenance = {
            "analysis": "F001 fixed-val inference ablation and fusion-state statistics",
            "checkpoint": portable_path(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "f001_formal_training_sha": F001_TRAINING_SHA,
            "analysis_git_sha": current_sha,
            "fixed_val_count": len(records),
            "deterministic_seed": args.seed,
            "imgsz": val_args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "modes": list(ABLATION_MODES),
            "fresh_model_per_mode": True,
            "fresh_model_for_statistics": True,
            "checkpoint_unchanged": sha256(checkpoint) == checkpoint_hash,
            "output_files": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
        }
        write_json(output_dir / "provenance.json", provenance)
    return provenance


def main() -> int:
    args = parse_args()
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
