"""Deterministic group construction and multilabel split optimization helpers."""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


CLASS_COUNT = 12
SIZE_CATEGORIES = ("tiny", "small", "medium", "large")
NUMERIC_MAX_GAP = 5
CORRELATION_THRESHOLD = 0.90
DHASH_THRESHOLD = 5


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        if root_left < root_right:
            self.parent[root_right] = root_left
        else:
            self.parent[root_left] = root_right


@dataclass(frozen=True)
class SampleGroup:
    group_id: str
    group_type: str
    stems: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.stems)


def filename_prefix(stem: str) -> str | None:
    tokens = stem.split("_")
    if len(tokens) >= 2 and tokens[-1].isdigit():
        return "_".join(tokens[:-1])
    return None


def build_groups(stems: list[str], pair_csv: Path) -> tuple[list[SampleGroup], dict[str, Any]]:
    stem_set = set(stems)
    union_find = UnionFind(stems)
    prefix_members: dict[str, list[str]] = defaultdict(list)
    for stem in stems:
        prefix = filename_prefix(stem)
        if prefix is not None:
            prefix_members[prefix].append(stem)
    prefix_edges = 0
    for members in prefix_members.values():
        anchor = min(members)
        for stem in members:
            if stem != anchor:
                union_find.union(anchor, stem)
                prefix_edges += 1

    numeric_visual_edges = 0
    with pair_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["pair_type"] != "numeric_adjacent":
                continue
            stem_a, stem_b = row["stem_a"], row["stem_b"]
            if stem_a not in stem_set or stem_b not in stem_set:
                continue
            frame_gap = int(row["frame_gap"])
            correlation = float(row["grayscale_correlation"])
            dhash_hamming = int(row["dhash_hamming"])
            if frame_gap <= NUMERIC_MAX_GAP and (
                correlation >= CORRELATION_THRESHOLD or dhash_hamming <= DHASH_THRESHOLD
            ):
                union_find.union(stem_a, stem_b)
                numeric_visual_edges += 1

    components: dict[str, list[str]] = defaultdict(list)
    for stem in stems:
        components[union_find.find(stem)].append(stem)
    ordered_components = sorted(
        (tuple(sorted(members)) for members in components.values()),
        key=lambda members: members[0],
    )
    groups: list[SampleGroup] = []
    for index, members in enumerate(ordered_components):
        prefixes = {filename_prefix(stem) for stem in members if filename_prefix(stem)}
        if len(members) == 1:
            group_type = "singleton"
        elif prefixes:
            group_type = "filename_prefix"
        else:
            group_type = "numeric_visual_sequence"
        groups.append(SampleGroup(f"G{index:04d}", group_type, members))

    metadata = {
        "group_count": len(groups),
        "multi_sample_group_count": sum(group.size > 1 for group in groups),
        "max_group_size": max((group.size for group in groups), default=0),
        "group_size_histogram": dict(
            sorted(Counter(group.size for group in groups).items())
        ),
        "rules": {
            "underscore_stems": "same prefix after removing final numeric frame token",
            "simple_numeric_stems": (
                "adjacent numeric pair with gap<=5 and "
                "(grayscale correlation>=0.90 or dHash Hamming<=5)"
            ),
            "prefix_union_edges": prefix_edges,
            "numeric_visual_union_edges": numeric_visual_edges,
        },
    }
    return groups, metadata


def build_feature_matrix(
    groups: list[SampleGroup], records: list[dict[str, Any]]
) -> tuple[np.ndarray, list[str]]:
    records_by_stem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_stem[record["stem"]].append(record)
    feature_names = (
        [f"class_object_{index}" for index in range(CLASS_COUNT)]
        + [f"class_image_{index}" for index in range(CLASS_COUNT)]
        + [f"size_{category}" for category in SIZE_CATEGORIES]
    )
    matrix = np.zeros((len(groups), len(feature_names)), dtype=np.float64)
    size_offset = CLASS_COUNT * 2
    for group_index, group in enumerate(groups):
        for stem in group.stems:
            stem_records = records_by_stem.get(stem, [])
            present_classes = {record["class_id"] for record in stem_records}
            for record in stem_records:
                matrix[group_index, record["class_id"]] += 1.0
                category_index = SIZE_CATEGORIES.index(record["size_category"])
                matrix[group_index, size_offset + category_index] += 1.0
            for class_id in present_classes:
                matrix[group_index, CLASS_COUNT + class_id] += 1.0
    return matrix, feature_names


def objective(value: np.ndarray, target: np.ndarray) -> float:
    relative_error = (value - target) / np.maximum(target, 1.0)
    return float(np.mean(relative_error * relative_error))


def select_validation_groups(
    groups: list[SampleGroup],
    features: np.ndarray,
    *,
    val_sample_count: int,
    val_fraction: float,
    seed: int,
    random_trials: int = 5000,
    local_iterations: int = 50000,
) -> tuple[set[int], dict[str, Any]]:
    rng = random.Random(seed)
    sizes = [group.size for group in groups]
    target = features.sum(axis=0) * val_fraction
    best_selected: set[int] | None = None
    best_value: np.ndarray | None = None
    best_score = float("inf")
    indices = list(range(len(groups)))

    for _ in range(random_trials):
        rng.shuffle(indices)
        remaining = val_sample_count
        selected: set[int] = set()
        value = np.zeros(features.shape[1], dtype=np.float64)
        for index in indices:
            if sizes[index] <= remaining:
                selected.add(index)
                value += features[index]
                remaining -= sizes[index]
                if remaining == 0:
                    break
        if remaining:
            continue
        score = objective(value, target)
        if score < best_score:
            best_selected = selected
            best_value = value.copy()
            best_score = score

    if best_selected is None or best_value is None:
        raise RuntimeError("Could not construct an exact-size group-aware validation split")

    selected = set(best_selected)
    value = best_value
    selected_by_size: dict[int, set[int]] = defaultdict(set)
    unselected_by_size: dict[int, set[int]] = defaultdict(set)
    for index, size in enumerate(sizes):
        (selected_by_size if index in selected else unselected_by_size)[size].add(index)
    common_sizes = sorted(set(selected_by_size) & set(unselected_by_size))
    accepted_swaps = 0
    for _ in range(local_iterations):
        size = rng.choice(common_sizes)
        out_index = rng.choice(tuple(selected_by_size[size]))
        in_index = rng.choice(tuple(unselected_by_size[size]))
        candidate_value = value - features[out_index] + features[in_index]
        candidate_score = objective(candidate_value, target)
        if candidate_score + 1e-15 < best_score:
            selected.remove(out_index)
            selected.add(in_index)
            selected_by_size[size].remove(out_index)
            selected_by_size[size].add(in_index)
            unselected_by_size[size].remove(in_index)
            unselected_by_size[size].add(out_index)
            value = candidate_value
            best_score = candidate_score
            accepted_swaps += 1

    metadata = {
        "objective": "mean squared relative deviation from 20% targets",
        "features": (
            "12 class object counts + 12 class image-presence counts + 4 size counts"
        ),
        "random_trials": random_trials,
        "local_swap_iterations": local_iterations,
        "accepted_local_swaps": accepted_swaps,
        "final_objective": best_score,
    }
    return selected, metadata
