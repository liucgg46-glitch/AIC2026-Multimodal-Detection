"""Machine-check CTRL001 training and augmentation parity with frozen F001."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.fusion.diagnostics import compare_ctrl001_to_f001  # noqa: E402


DEFAULT_CTRL = ROOT / "configs/experiments/CTRL001_RGB_F001_AUG.yaml"
DEFAULT_F001 = ROOT / "configs/experiments/F001_RGB_IR_GATED_P45_YOLO11N.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ctrl", type=Path, default=DEFAULT_CTRL)
    parser.add_argument("--f001", type=Path, default=DEFAULT_F001)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = compare_ctrl001_to_f001(args.ctrl, args.f001)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise RuntimeError("CTRL001 differs from F001 outside approved RGB/model fields")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
