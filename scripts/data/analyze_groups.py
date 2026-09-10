"""Audit possible sequence/group leakage before freezing train/val splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_DIR = REPO_ROOT / "data" / "raw" / "train" / "visible"
DEFAULT_SPLIT_DIR = REPO_ROOT / "data" / "splits"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SEED = 2026


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare candidate sequence pairs with random image pairs."
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def naming_pattern(stem: str) -> str:
    if stem.isdigit():
        return "simple_numeric"
    return re.sub(r"\d+", "{N}", stem)


def prefix_group(stem: str) -> str | None:
    tokens = stem.split("_")
    if len(tokens) >= 2 and tokens[-1].isdigit():
        return "_".join(tokens[:-1])
    return None


def final_index(stem: str) -> int | None:
    token = stem.split("_")[-1]
    return int(token) if token.isdigit() else None


def read_split(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def load_features(image_paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    for stem, path in tqdm(
        sorted(image_paths.items()), desc="Extracting image features", unit="file"
    ):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        small -= float(small.mean())
        norm = float(np.linalg.norm(small))
        normalized = small.ravel() / norm if norm else small.ravel()
        hash_image = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        bits = hash_image[:, 1:] > hash_image[:, :-1]
        hash_value = int.from_bytes(np.packbits(bits).tobytes(), "big")
        features[stem] = {"vector": normalized, "dhash": hash_value}
    return features


def split_relation(stem_a: str, stem_b: str, train: set[str], val: set[str]) -> str:
    if stem_a in train and stem_b in train:
        return "both_train"
    if stem_a in val and stem_b in val:
        return "both_val"
    if (stem_a in train and stem_b in val) or (stem_a in val and stem_b in train):
        return "cross_split"
    return "split_unavailable"


def pair_record(
    pair_type: str,
    stem_a: str,
    stem_b: str,
    features: dict[str, dict[str, Any]],
    train: set[str],
    val: set[str],
) -> dict[str, Any]:
    index_a, index_b = final_index(stem_a), final_index(stem_b)
    frame_gap = abs(index_b - index_a) if index_a is not None and index_b is not None else ""
    correlation = float(np.dot(features[stem_a]["vector"], features[stem_b]["vector"]))
    hamming = int((features[stem_a]["dhash"] ^ features[stem_b]["dhash"]).bit_count())
    return {
        "pair_type": pair_type,
        "stem_a": stem_a,
        "stem_b": stem_b,
        "frame_gap": frame_gap,
        "grayscale_correlation": correlation,
        "dhash_hamming": hamming,
        "split_relation": split_relation(stem_a, stem_b, train, val),
    }


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "max": float(array.max()),
    }


def main() -> int:
    args = parse_args()
    image_dir = args.image_dir.resolve()
    split_dir = args.split_dir.resolve()
    output_dir = args.output_dir.resolve()
    image_paths = {
        path.stem: path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    }
    if len(image_paths) != 2000:
        raise ValueError(f"Expected 2000 visible images, found {len(image_paths)}")

    patterns = Counter(naming_pattern(stem) for stem in image_paths)
    groups: dict[str, list[str]] = defaultdict(list)
    numeric_stems: list[str] = []
    for stem in image_paths:
        group = prefix_group(stem)
        if group is not None:
            groups[group].append(stem)
        elif stem.isdigit():
            numeric_stems.append(stem)

    multi_groups = {
        group: sorted(stems, key=lambda stem: final_index(stem) or -1)
        for group, stems in groups.items()
        if len(stems) > 1
    }
    prefix_pairs = [
        (stems[index], stems[index + 1])
        for stems in multi_groups.values()
        for index in range(len(stems) - 1)
    ]
    numeric_sorted = sorted(numeric_stems, key=int)
    numeric_pairs = list(zip(numeric_sorted, numeric_sorted[1:]))

    features = load_features(image_paths)
    train = read_split(split_dir / "train.txt")
    val = read_split(split_dir / "val.txt")
    rows: list[dict[str, Any]] = []
    for stem_a, stem_b in prefix_pairs:
        rows.append(pair_record("same_prefix_adjacent", stem_a, stem_b, features, train, val))
    for stem_a, stem_b in numeric_pairs:
        rows.append(pair_record("numeric_adjacent", stem_a, stem_b, features, train, val))

    rng = random.Random(SEED)
    stems = sorted(image_paths)
    random_pairs: set[tuple[str, str]] = set()
    target_count = max(len(prefix_pairs), len(numeric_pairs))
    while len(random_pairs) < target_count:
        stem_a, stem_b = rng.sample(stems, 2)
        pair = tuple(sorted((stem_a, stem_b)))
        if prefix_group(stem_a) == prefix_group(stem_b) and prefix_group(stem_a) is not None:
            continue
        random_pairs.add(pair)
    for stem_a, stem_b in sorted(random_pairs):
        rows.append(pair_record("random_baseline", stem_a, stem_b, features, train, val))

    pair_types: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pair_types[row["pair_type"]].append(row)
    pair_summary = {}
    for pair_type, type_rows in pair_types.items():
        pair_summary[pair_type] = {
            "grayscale_correlation": summarize(
                [float(row["grayscale_correlation"]) for row in type_rows]
            ),
            "dhash_hamming": summarize(
                [float(row["dhash_hamming"]) for row in type_rows]
            ),
            "cross_split_pair_count": sum(
                row["split_relation"] == "cross_split" for row in type_rows
            ),
        }

    group_split_counts = Counter()
    crossing_groups: list[dict[str, Any]] = []
    for group, stems_in_group in sorted(multi_groups.items()):
        train_count = sum(stem in train for stem in stems_in_group)
        val_count = sum(stem in val for stem in stems_in_group)
        relation = "cross_split" if train_count and val_count else "single_split"
        group_split_counts[relation] += 1
        if relation == "cross_split":
            crossing_groups.append(
                {
                    "group": group,
                    "sample_count": len(stems_in_group),
                    "train_count": train_count,
                    "val_count": val_count,
                    "stems": stems_in_group,
                }
            )

    summary = {
        "image_count": len(image_paths),
        "seed": SEED,
        "naming_patterns": dict(sorted(patterns.items())),
        "simple_numeric_stem_count": len(numeric_stems),
        "prefix_extractable_sample_count": sum(len(stems_in_group) for stems_in_group in groups.values()),
        "prefix_group_count": len(groups),
        "multi_sample_prefix_group_count": len(multi_groups),
        "multi_sample_prefix_group_size_histogram": dict(
            sorted(Counter(len(stems_in_group) for stems_in_group in multi_groups.values()).items())
        ),
        "candidate_pair_statistics": pair_summary,
        "current_split_group_audit": {
            "multi_sample_groups": len(multi_groups),
            "cross_split_groups": group_split_counts["cross_split"],
            "single_split_groups": group_split_counts["single_split"],
            "crossing_groups": crossing_groups,
        },
        "interpretation_limit": (
            "Filename prefix and low-resolution visual similarity provide evidence, not official "
            "sequence metadata. Pure numeric stems cannot be assigned to scene groups from names alone."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "group_audit_summary.json"
    pairs_path = output_dir / "group_candidate_pairs.csv"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pairs_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("Group leakage audit complete")
    print(f"  images: {len(image_paths)}")
    print(f"  multi-sample prefix groups: {len(multi_groups)}")
    print(f"  groups crossing current split: {group_split_counts['cross_split']}")
    print(f"  summary: {summary_path}")
    print(f"  pair details: {pairs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
