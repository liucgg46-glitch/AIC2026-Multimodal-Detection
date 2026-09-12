"""Validate S003 predictions, report statistics, optionally create a verified TXT ZIP."""
import argparse
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("submission_format", ROOT / "tests/test_submission_format.py")
FORMAT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FORMAT)
EXPECTED_COUNT = 1000


def prepare(images, predictions, zip_output=None):
    images, predictions = Path(images), Path(predictions)
    errors = FORMAT.validate_submission(images, predictions, max_det=100)
    if errors:
        raise ValueError("\n".join(errors))
    stems, _ = FORMAT.image_stems(images)
    if len(stems) != EXPECTED_COUNT or len({s.casefold() for s in stems}) != EXPECTED_COUNT:
        raise ValueError("Expected 1000 unique casefold image stems")
    files = sorted(predictions.iterdir(), key=lambda p: p.name)
    if len(files) != EXPECTED_COUNT or any(not p.is_file() or p.is_symlink() or p.suffix != ".txt" for p in files):
        raise ValueError("Prediction directory must contain exactly 1000 regular TXT files only")
    contents = {p.name: p.read_bytes() for p in files}
    counts = [len([line for line in data.decode("utf-8-sig").splitlines() if line.strip()])
              for data in contents.values()]
    result = dict(input_images=len(stems), txt_files=len(files), missing=0, extra=0,
                  nonempty=sum(n > 0 for n in counts), empty=sum(n == 0 for n in counts),
                  total_boxes=sum(counts), mean_boxes_per_image=sum(counts) / len(counts),
                  max_boxes_per_image=max(counts), hit_max_det_100=sum(n == 100 for n in counts),
                  class_id_valid=True, normalized_xywh_valid=True, confidence_valid=True,
                  nan_inf_count=0)
    if zip_output is not None:
        zip_output = Path(zip_output)
        if zip_output.suffix.lower() != ".zip":
            raise ValueError("ZIP output must end in .zip")
        if predictions.resolve() == zip_output.resolve().parent or predictions.resolve() in zip_output.resolve().parents:
            raise ValueError("ZIP must be outside the TXT directory")
        # Exclusive creation: never overwrite any existing file.
        with zipfile.ZipFile(zip_output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in contents.items():
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, data)
        with zipfile.ZipFile(zip_output) as archive:
            names = archive.namelist()
            if len(names) != EXPECTED_COUNT or set(names) != set(contents) or archive.testzip() is not None:
                raise ValueError("ZIP member/CRC mismatch")
            if any(archive.read(name) != data for name, data in contents.items()):
                raise ValueError("ZIP content mismatch")
        result["zip_members"] = len(names)
        result["zip_sha256"] = hashlib.sha256(zip_output.read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default="data/raw/test/depth")
    parser.add_argument("--predictions", default="outputs/submissions/E003_DEPTH_YOLO11N_CLEAN_PRELIM_001")
    parser.add_argument("--zip-output", help="Optional new ZIP outside the TXT directory; no overwrite")
    args = parser.parse_args()
    try:
        result = prepare(FORMAT.project_path(args.images), FORMAT.project_path(args.predictions),
                         FORMAT.project_path(args.zip_output) if args.zip_output else None)
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
