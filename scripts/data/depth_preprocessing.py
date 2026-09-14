"""Deterministic, PNG-only mathematical Depth preprocessing candidates for C3."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_DEPTH_RELATIVE = Path("data/raw/train/depth")
CANONICAL_TRAIN_SPLIT_RELATIVE = Path("data/splits/train.txt")
CANONICAL_VAL_SPLIT_RELATIVE = Path("data/splits/val.txt")
CANONICAL_TRAIN_NORMALIZED_SHA256 = "20b0c1fb09a6848a4700a5adc8ce1f9a6d9040a48626a1759020d7f986450969"
CANONICAL_VAL_NORMALIZED_SHA256 = "336165b3509b6a0516b052c02fd98d152441300e8ae757eb0d27ca68a053ee48"
EXPECTED_TRAIN_COUNT = 1600
EXPECTED_VAL_COUNT = 400
EXPECTED_TRAIN_PNG_COUNT = 1478
EXPECTED_TRAIN_JPG_COUNT = 122
LOW_PERCENTILE = 2
HIGH_PERCENTILE = 98
VALID_CANDIDATES = ("validmask", "percentile", "log", "inverse")
QUANTILE_METHOD = "uint16_histogram_nearest_rank_1based_ceil"


def read_image_unchanged(path):
    """Decode original bit depth/shape, independent of Ultralytics' cv2.imread patch."""
    encoded = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    return cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED) if encoded.size else None


class DepthPreprocessingError(ValueError):
    """Raised when the C3 input or frozen preprocessing contract is invalid."""


@dataclass(frozen=True)
class PercentileParams:
    """Frozen train-PNG-only parameters for the percentile candidate."""

    candidate: str
    p_low: int
    p_high: int
    low_value: int
    high_value: int
    valid_pixel_count: int
    method: str
    zero_excluded: bool
    source_scope: str
    png_count: int
    train_split_normalized_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def identity_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CandidateResult:
    """A transformed array, its independent validity mask, and stable metadata."""

    values: np.ndarray
    valid_mask: np.ndarray
    metadata: Dict[str, Any]


def normalize_split_content(content: Union[str, bytes]) -> bytes:
    """Normalize newline encoding only and serialize with one final terminator."""
    if isinstance(content, bytes):
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DepthPreprocessingError("Split 必须是 UTF-8 文本") from exc
    else:
        text = content.lstrip("\ufeff")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    logical_lines = text.split("\n")
    if text.endswith("\n"):
        logical_lines = logical_lines[:-1]
    return ("\n".join(logical_lines) + "\n").encode("utf-8")


def normalized_split_sha256(path: Path) -> str:
    return hashlib.sha256(normalize_split_content(path.read_bytes())).hexdigest()


def _validate_png_depth(depth: np.ndarray, source_format: str = "png") -> None:
    if source_format.lower() != "png":
        raise DepthPreprocessingError(
            "C3 physical Depth candidates only support PNG uint16 source; JPG physical mapping is unknown"
        )
    if not isinstance(depth, np.ndarray):
        raise DepthPreprocessingError("Depth input 必须是 numpy.ndarray")
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise DepthPreprocessingError(
            f"C3 只接受 IMREAD_UNCHANGED 的 uint16 2D PNG Depth；实际 dtype={depth.dtype}, shape={depth.shape}"
        )


def make_valid_mask(depth: np.ndarray, source_format: str = "png") -> np.ndarray:
    """Return a new bool mask using the canonical PNG rule: valid = depth > 0."""
    _validate_png_depth(depth, source_format)
    return np.greater(depth, 0)


def _nearest_rank_value(histogram: np.ndarray, valid_count: int, percentile: int) -> int:
    """Return the 1-based nearest rank value where rank=ceil(percentile/100*N)."""
    if valid_count <= 0:
        raise DepthPreprocessingError("没有 valid Depth pixel，无法拟合 percentile")
    rank = max(1, (percentile * valid_count + 99) // 100)
    cumulative = np.cumsum(histogram, dtype=np.uint64)
    return int(np.searchsorted(cumulative, rank, side="left"))


def fit_train_png_percentiles(
    png_paths: Sequence[Path],
    *,
    expected_count: Optional[int] = None,
    train_split_normalized_sha256: str = "",
) -> PercentileParams:
    """Fit P2/P98 from nonzero pixels using an exact uint16 histogram.

    For N pooled valid pixels, percentile p uses the 1-based nearest rank
    ``rank(p) = ceil(p / 100 * N)``. The selected value is the first uint16 bin
    whose cumulative count is at least that rank. Zeros never enter the histogram.
    """
    paths = [Path(path) for path in png_paths]
    if expected_count is not None and len(paths) != expected_count:
        raise DepthPreprocessingError(
            f"train PNG 数量错误：期望 {expected_count}，实际 {len(paths)}"
        )
    identities = [os.path.normcase(str(path.resolve())) for path in paths]
    if len(set(identities)) != len(identities):
        raise DepthPreprocessingError("train PNG paths 包含重复或同一 resolved 文件")
    histogram = np.zeros(65536, dtype=np.uint64)
    for path in paths:
        if path.suffix.lower() != ".png":
            raise DepthPreprocessingError(f"Percentile fit 只接受 train PNG，不接受: {path.name}")
        depth = read_image_unchanged(path)
        if depth is None:
            raise DepthPreprocessingError(f"PNG Depth 解码失败: {path}")
        _validate_png_depth(depth)
        counts = np.bincount(depth.reshape(-1), minlength=65536).astype(np.uint64)
        counts[0] = 0
        histogram += counts
    valid_count = int(histogram.sum())
    low_value = _nearest_rank_value(histogram, valid_count, LOW_PERCENTILE)
    high_value = _nearest_rank_value(histogram, valid_count, HIGH_PERCENTILE)
    if high_value <= low_value:
        raise DepthPreprocessingError(
            f"Percentile 分布退化：P{LOW_PERCENTILE}={low_value}, P{HIGH_PERCENTILE}={high_value}"
        )
    return PercentileParams(
        candidate="percentile",
        p_low=LOW_PERCENTILE,
        p_high=HIGH_PERCENTILE,
        low_value=low_value,
        high_value=high_value,
        valid_pixel_count=valid_count,
        method=QUANTILE_METHOD,
        zero_excluded=True,
        source_scope="train_png_only",
        png_count=len(paths),
        train_split_normalized_sha256=train_split_normalized_sha256,
    )


def _read_split(path: Path, expected_count: int) -> List[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(lines) != expected_count:
        raise DepthPreprocessingError(
            f"Split 数量错误：{path}，期望 {expected_count}，实际 {len(lines)}"
        )
    folded = [line.casefold() for line in lines]
    if len(set(folded)) != len(folded):
        raise DepthPreprocessingError(f"Split 包含重复或仅大小写不同的 stem: {path}")
    if any(Path(line).name != line or Path(line).suffix for line in lines):
        raise DepthPreprocessingError(f"Split 包含非法 stem: {path}")
    return lines


def _index_depth_files(depth_dir: Path) -> Dict[str, Path]:
    if not depth_dir.is_dir():
        raise DepthPreprocessingError(f"Canonical Depth 目录不存在: {depth_dir}")
    grouped: Dict[str, List[Path]] = {}
    for path in depth_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            grouped.setdefault(path.stem.casefold(), []).append(path)
    ambiguous = [paths for paths in grouped.values() if len(paths) != 1]
    if ambiguous:
        raise DepthPreprocessingError(
            f"Depth 存在 duplicate/casefold stem: {[path.name for path in ambiguous[0]]}"
        )
    return {identity: paths[0] for identity, paths in grouped.items()}


def fit_canonical_train_png_percentiles() -> PercentileParams:
    """Fit only the 1478 PNG files named by the canonical fixed train split."""
    root = PROJECT_ROOT.resolve()
    depth_dir = root / CANONICAL_DEPTH_RELATIVE
    train_split = root / CANONICAL_TRAIN_SPLIT_RELATIVE
    val_split = root / CANONICAL_VAL_SPLIT_RELATIVE
    if normalized_split_sha256(train_split) != CANONICAL_TRAIN_NORMALIZED_SHA256:
        raise DepthPreprocessingError("Canonical train split normalized identity 不一致")
    if normalized_split_sha256(val_split) != CANONICAL_VAL_NORMALIZED_SHA256:
        raise DepthPreprocessingError("Canonical val split normalized identity 不一致")
    train_stems = _read_split(train_split, EXPECTED_TRAIN_COUNT)
    val_stems = _read_split(val_split, EXPECTED_VAL_COUNT)
    if set(stem.casefold() for stem in train_stems) & set(stem.casefold() for stem in val_stems):
        raise DepthPreprocessingError("Canonical train/val split 存在 overlap")
    depth_files = _index_depth_files(depth_dir)
    missing = [stem for stem in train_stems if stem.casefold() not in depth_files]
    if missing:
        raise DepthPreprocessingError(f"Train split 缺少 Depth 文件: {missing[:5]}")
    train_files = [depth_files[stem.casefold()] for stem in train_stems]
    png_paths = [path for path in train_files if path.suffix.lower() == ".png"]
    jpg_count = sum(path.suffix.lower() in {".jpg", ".jpeg"} for path in train_files)
    if len(png_paths) != EXPECTED_TRAIN_PNG_COUNT or jpg_count != EXPECTED_TRAIN_JPG_COUNT:
        raise DepthPreprocessingError(
            "Canonical train PNG/JPG 数量错误："
            f"期望 {EXPECTED_TRAIN_PNG_COUNT}/{EXPECTED_TRAIN_JPG_COUNT}，"
            f"实际 {len(png_paths)}/{jpg_count}"
        )
    return fit_train_png_percentiles(
        png_paths,
        expected_count=EXPECTED_TRAIN_PNG_COUNT,
        train_split_normalized_sha256=CANONICAL_TRAIN_NORMALIZED_SHA256,
    )


def _validate_percentile_params(params: PercentileParams) -> None:
    if not isinstance(params, PercentileParams) or params.candidate != "percentile":
        raise DepthPreprocessingError("Percentile candidate 必须提供 fitted PercentileParams")
    if (
        params.p_low != LOW_PERCENTILE
        or params.p_high != HIGH_PERCENTILE
        or params.method != QUANTILE_METHOD
        or not params.zero_excluded
        or params.source_scope != "train_png_only"
        or params.valid_pixel_count <= 0
        or params.png_count <= 0
    ):
        raise DepthPreprocessingError("PercentileParams 不符合冻结的 C3 P2/P98 合同")
    if params.high_value <= params.low_value:
        raise DepthPreprocessingError("PercentileParams 要求 high_value > low_value")


def apply_percentile(
    depth: np.ndarray,
    params: PercentileParams,
    source_format: str = "png",
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply frozen train-PNG P2/P98 clipping and return float32 values plus mask."""
    _validate_png_depth(depth, source_format)
    _validate_percentile_params(params)
    valid = make_valid_mask(depth)
    output = np.zeros(depth.shape, dtype=np.float32)
    values = depth[valid].astype(np.float32)
    output[valid] = np.clip(
        (values - np.float32(params.low_value))
        / np.float32(params.high_value - params.low_value),
        np.float32(0.0),
        np.float32(1.0),
    )
    if not np.isfinite(output).all():
        raise DepthPreprocessingError("Percentile output 包含 NaN/Inf")
    return output, valid


def apply_log_depth(
    depth: np.ndarray, source_format: str = "png"
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply natural log1p to valid raw millimeter Depth values only."""
    _validate_png_depth(depth, source_format)
    valid = make_valid_mask(depth)
    output = np.zeros(depth.shape, dtype=np.float32)
    output[valid] = np.log1p(depth[valid].astype(np.float32))
    if not np.isfinite(output).all():
        raise DepthPreprocessingError("Log output 包含 NaN/Inf")
    return output, valid


def apply_inverse_depth(
    depth: np.ndarray, source_format: str = "png"
) -> Tuple[np.ndarray, np.ndarray]:
    """Return reciprocal millimeter Depth for valid pixels, without clipping or scale."""
    _validate_png_depth(depth, source_format)
    valid = make_valid_mask(depth)
    output = np.zeros(depth.shape, dtype=np.float32)
    output[valid] = np.float32(1.0) / depth[valid].astype(np.float32)
    if not np.isfinite(output).all():
        raise DepthPreprocessingError("Inverse output 包含 NaN/Inf")
    return output, valid


def candidate_metadata(
    candidate: str, percentile_params: Optional[PercentileParams] = None
) -> Dict[str, Any]:
    if candidate not in VALID_CANDIDATES:
        raise DepthPreprocessingError(f"Unknown C3 candidate: {candidate}")
    if candidate != "percentile" and percentile_params is not None:
        raise DepthPreprocessingError("PercentileParams 只能用于 percentile candidate")
    common = {
        "candidate": candidate,
        "input_dtype": "uint16",
        "physical_scope": "PNG physical depth only; JPG unsupported",
    }
    if candidate == "validmask":
        return dict(common, output_dtype="bool", valid_rule="depth > 0")
    if candidate == "percentile":
        if percentile_params is None:
            raise DepthPreprocessingError("Percentile candidate requires fitted parameters")
        _validate_percentile_params(percentile_params)
        parameter_fields = percentile_params.to_dict()
        return dict(
            common,
            output_dtype="float32",
            output_range=[0.0, 1.0],
            p_low=parameter_fields["p_low"],
            p_high=parameter_fields["p_high"],
            low_value=parameter_fields["low_value"],
            high_value=parameter_fields["high_value"],
            valid_pixel_count=parameter_fields["valid_pixel_count"],
            quantile_method=parameter_fields["method"],
            fit_scope=parameter_fields["source_scope"],
            png_count=parameter_fields["png_count"],
            train_split_normalized_sha256=parameter_fields[
                "train_split_normalized_sha256"
            ],
            zero_policy="exclude zero from fit; preserve invalid zero in output",
            parameter_identity_sha256=percentile_params.identity_sha256(),
            valid_rule="depth > 0; transformed zero alone does not identify invalid",
        )
    if candidate == "log":
        return dict(
            common,
            formula="natural_log1p_depth",
            input_unit="mm",
            invalid_rule="depth == 0",
            invalid_output=0,
            output_dtype="float32",
        )
    return dict(
        common,
        formula="reciprocal_depth",
        input_unit="mm",
        invalid_rule="depth == 0",
        invalid_output=0,
        output_dtype="float32",
        output_unit="mm^-1",
    )


def apply_candidate(
    depth: np.ndarray,
    candidate: str,
    percentile_params: Optional[PercentileParams] = None,
    source_format: str = "png",
) -> CandidateResult:
    """Apply exactly one named C3 candidate; lists and pipelines are not accepted."""
    if not isinstance(candidate, str) or candidate not in VALID_CANDIDATES:
        raise DepthPreprocessingError(f"Unknown or combined C3 candidate: {candidate}")
    if candidate != "percentile" and percentile_params is not None:
        raise DepthPreprocessingError("PercentileParams 只能用于 percentile candidate")
    if candidate == "validmask":
        mask = make_valid_mask(depth, source_format)
        values = mask.copy()
    elif candidate == "percentile":
        if percentile_params is None:
            raise DepthPreprocessingError("Percentile candidate requires fitted parameters")
        values, mask = apply_percentile(depth, percentile_params, source_format)
    elif candidate == "log":
        values, mask = apply_log_depth(depth, source_format)
    else:
        values, mask = apply_inverse_depth(depth, source_format)
    return CandidateResult(
        values=values,
        valid_mask=mask,
        metadata=candidate_metadata(candidate, percentile_params),
    )


def canonical_metadata_json(metadata: Dict[str, Any]) -> str:
    """Serialize candidate metadata deterministically without writing a file."""
    return json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
