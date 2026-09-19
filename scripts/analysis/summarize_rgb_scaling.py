"""Summarize completed RGB scaling runs into one portable JSON report."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(row: Dict[str, str], key: str) -> float:
    return float(row[key].strip())


def summarize(run: Path) -> Dict[str, Any]:
    results = run / "results.csv"
    args = run / "args.yaml"
    best_weight = run / "weights" / "best.pt"
    missing = [str(path) for path in (results, args, best_weight) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Run artifact incomplete: " + ", ".join(missing))
    with results.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = [{key.strip(): value for key, value in row.items()} for row in csv.DictReader(stream)]
    if not rows:
        raise ValueError("Empty results.csv: " + str(results))
    metric = "metrics/mAP50-95(B)"
    best = max(rows, key=lambda row: number(row, metric))
    last = rows[-1]
    return {
        "run": run.name,
        "epochs_completed": len(rows),
        "best_epoch": int(number(best, "epoch")),
        "best_map50_95": number(best, metric),
        "best_map50": number(best, "metrics/mAP50(B)"),
        "best_precision": number(best, "metrics/precision(B)"),
        "best_recall": number(best, "metrics/recall(B)"),
        "last_map50_95": number(last, metric),
        "best_weight_sha256": sha256(best_weight),
        "results_sha256": sha256(results),
        "args_sha256": sha256(args),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report: List[Dict[str, Any]] = [summarize(path.resolve()) for path in args.runs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump({"runs": report}, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"runs": report}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
