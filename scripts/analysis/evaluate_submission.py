"""Local TXT evaluator following the supplied AIC PDF, not an official scoring script.

Confidence-ordered greedy matching, per-image global top 100, arithmetic mean
of the 101 precision-envelope samples. Ties use filename/line order; matching
chooses the highest-IoU unmatched GT. The PDF does not specify those tie rules.
No clipping of GT boxes: preserve the supplied annotation geometry.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read_rows(path, prediction=False):
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = np.asarray([float(x) for x in line.split()], dtype=np.float64)
        if (len(row) != (6 if prediction else 5) or not np.isfinite(row).all()
                or row[0] != int(row[0]) or not 0 <= row[0] < 12
                or (row[1:] < 0).any() or (row[1:] > 1).any()
                or row[3] <= 0 or row[4] <= 0):
            raise ValueError("Invalid TXT row: %s:%s" % (path, number))
        rows.append(row.tolist())
    return rows


def ious(box, boxes):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    box = np.asarray(box, dtype=np.float64)
    lo = np.maximum(box[:2] - box[2:] / 2, boxes[:, :2] - boxes[:, 2:] / 2)
    hi = np.minimum(box[:2] + box[2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2)
    intersection = np.maximum(hi - lo, 0).prod(axis=1)
    return intersection / (box[2:].prod() + boxes[:, 2:].prod(axis=1) - intersection)


def ap101(tp, gt_count):
    if not gt_count or not len(tp):
        return 0.0
    cumulative = np.cumsum(tp, dtype=np.float64)
    recall = cumulative / gt_count
    precision = cumulative / np.arange(1, len(tp) + 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    indices = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    return float(np.append(envelope, 0)[indices].mean())


def evaluate(ground_truth, predictions, conf=0.001, max_det=100):
    if set(ground_truth) != set(predictions):
        raise ValueError("Prediction files must exactly match validation stems, including empty TXT")
    if not 0 <= conf <= 1 or not 1 <= max_det <= 100:
        raise ValueError("Invalid conf or max_det")
    predictions = {stem: sorted((r for r in rows if r[5] >= conf), key=lambda r: -r[5])[:max_det]
                   for stem, rows in predictions.items()}
    classes = []
    for cls in range(12):
        gt = {s: [r[1:5] for r in rows if int(r[0]) == cls] for s, rows in ground_truth.items()}
        count = sum(map(len, gt.values()))
        ranked = [(r[5], s, i, r[1:5]) for s in sorted(predictions)
                  for i, r in enumerate(predictions[s]) if int(r[0]) == cls]
        ranked.sort(key=lambda r: (-r[0], r[1], r[2]))
        aps = []
        for threshold in np.linspace(0.5, 0.95, 10):
            used = {s: np.zeros(len(rows), dtype=bool) for s, rows in gt.items()}
            tp = []
            for _, stem, _, box in ranked:
                overlap = ious(box, gt[stem])
                overlap[used[stem]] = -1
                best = int(overlap.argmax()) if len(overlap) else -1
                matched = best >= 0 and overlap[best] >= threshold
                if matched:
                    used[stem][best] = True
                tp.append(matched)
            aps.append(ap101(tp, count))
        classes.append(dict(class_id=cls, gt=count, predictions=len(ranked), ap50=aps[0],
                            ap50_95=float(np.mean(aps)), ap_by_iou=aps))
    return dict(evaluator="AIC PDF reconstruction; not official scorer", images=len(ground_truth),
                conf=conf, max_det=max_det, absent_class_policy="zero; average all 12 classes",
                map50=float(np.mean([r["ap50"] for r in classes])),
                map50_95=float(np.mean([r["ap50_95"] for r in classes])), per_class=classes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conf", type=float, nargs="+", default=[0.001, 0.01, 0.05, 0.25])
    args = parser.parse_args()
    stems = (ROOT / "data/splits/val.txt").read_text(encoding="utf-8-sig").split()
    if len(stems) != 400 or len(set(stems)) != 400:
        raise ValueError("Expected fixed validation split of 400 unique stems")
    files = list(args.predictions.glob("*.txt"))
    if {p.stem for p in files} != set(stems):
        raise ValueError("Expected exactly fixed val TXT files; test submissions are not validation inputs")
    gt_dir = ROOT / "data/processed/train/labels_clean"
    gt = {s: read_rows(gt_dir / (s + ".txt")) for s in stems}
    pred = {s: read_rows(args.predictions / (s + ".txt"), True) for s in stems}
    digest = hashlib.sha256()
    for s in sorted(stems):
        digest.update(s.encode("utf-8"))
        for folder in (gt_dir, args.predictions):
            digest.update(hashlib.sha256((folder / (s + ".txt")).read_bytes()).digest())
    report = dict(input_sha256=digest.hexdigest(),
                  note="Sweep only removes candidates; export initially at conf=0.001 or lower.",
                  results=[evaluate(gt, pred, conf=c) for c in args.conf])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    for r in report["results"]:
        print("conf=%.4f mAP50-95=%.6f" % (r["conf"], r["map50_95"]))


if __name__ == "__main__":
    main()
