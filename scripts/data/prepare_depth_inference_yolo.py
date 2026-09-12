"""Stage the 1000 prelim-test Depth images with the frozen C4 inverse writer."""
import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_depth_candidate_yolo as c4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("data/raw/test/depth")
OUTPUT = Path("data/processed/depth_inference/inverse/prelim_test")
EXPECTED_COUNT = 1000
EXTENSIONS = {".png", ".jpg", ".jpeg"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def absolute(path):
    path = Path(path)
    return Path(os.path.abspath(str(path if path.is_absolute() else PROJECT_ROOT / path)))


def check_paths(source, output):
    require(source == absolute(SOURCE), "Source must be canonical data/raw/test/depth")
    require(output == absolute(OUTPUT), "Output must be canonical Depth inference view")
    root = PROJECT_ROOT.resolve()
    # Check lexical ancestry against an independent project anchor, including junctions.
    for leaf in (source, output):
        for path in (leaf, *leaf.parents):
            if path == PROJECT_ROOT.parent:
                break
            expected = root / path.relative_to(PROJECT_ROOT)
            require(not path.is_symlink() and path.resolve() == expected,
                    "Redirected source/output ancestor: {}".format(path))
    require(source.is_dir(), "Source directory missing")
    require(not output.exists(), "Output already exists; overwrite is forbidden")


def source_inventory(source):
    files = sorted(source.iterdir(), key=lambda p: (p.name.casefold(), p.name))
    require(all(p.is_file() and not p.is_symlink() for p in files), "Source must contain regular files only")
    require(all(p.name == ".gitkeep" or p.suffix.lower() in EXTENSIONS for p in files),
            "Only PNG/JPG images (and .gitkeep) are supported")
    return {p.name: c4.sha256_file(p) for p in files}


def identity(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def build_view(source, output):
    source, output = absolute(source), absolute(output)
    check_paths(source, output)
    before = source_inventory(source)
    images = [source / name for name in before if name != ".gitkeep"]
    require(len(images) == EXPECTED_COUNT, "Expected exactly 1000 test images")
    require(len({p.stem.casefold() for p in images}) == len(images), "Duplicate/casefold stem conflict")
    output.parent.mkdir(parents=True, exist_ok=True)
    check_paths(source, output)
    staging = Path(tempfile.mkdtemp(prefix=".prelim_test-", dir=str(output.parent)))
    # Failed builds retain their private temporary directory for inspection; no recursive deletion.
    (staging / "images").mkdir()
    records = []
    for image in images:
        check_paths(source, output)
        require(staging.resolve().parent == output.parent.resolve() and not staging.is_symlink(),
                "Temporary staging redirected")
        destination = staging / "images" / (image.stem + image.suffix.lower())
        decoded_source = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        require(decoded_source is not None, "Source cannot be decoded: {}".format(image.name))
        if image.suffix.lower() == ".png":
            c4.write_candidate_png(image, destination, "inverse", None)
        else:
            require(decoded_source.dtype == np.uint8 and decoded_source.ndim == 3
                    and decoded_source.shape[2] == 3, "JPG must be uint8 x3")
            c4.copy_isolated(image, destination)
        decoded = cv2.imread(str(destination), cv2.IMREAD_UNCHANGED)
        require(decoded is not None and decoded.dtype == np.uint8 and decoded.ndim == 3
                and decoded.shape[2] == 3, "Output must decode as uint8 x3")
        require(decoded.shape[:2] == decoded_source.shape[:2], "Geometry changed")
        digest = c4.sha256_file(destination)
        if image.suffix.lower() != ".png":
            require(digest == before[image.name], "JPG bytes changed")
        records.append({"stem": image.stem, "source": (SOURCE / image.name).as_posix(),
                        "output": "images/" + destination.name,
                        "source_sha256": before[image.name], "output_sha256": digest})
    require(source_inventory(source) == before, "Source changed during staging")
    require({p.name for p in (staging / "images").iterdir()} ==
            {r["output"].split("/")[-1] for r in records}, "Output image set mismatch")
    counts = Counter("png" if p.suffix.lower() == ".png" else "jpg" for p in images)
    manifest = {
        "schema_version": 1, "representation": "inverse", "purpose": "prelim_test_inference",
        "source_path": SOURCE.as_posix(), "output_path": OUTPUT.as_posix(),
        "total": len(records), "png_count": counts["png"], "jpg_count": counts["jpg"],
        "unique_casefold_stems": len(records), "source_unchanged": True,
        "output_dtype": "uint8", "channels": 3,
        "source_identity_sha256": identity(before),
        "output_identity_sha256": identity({r["output"]: r["output_sha256"] for r in records}),
        "conversion": c4.candidate_conversion_metadata("inverse", None),
        "jpg_strategy": "byte-preserving passthrough", "jpg_physical_unit": "unknown",
        "files": records,
    }
    with (staging / "manifest.json").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    check_paths(source, output)
    require(staging.resolve().parent == output.parent.resolve() and not staging.is_symlink(),
            "Temporary staging redirected")
    staging.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=["inverse"], default="inverse")
    parser.add_argument("--source", default=SOURCE.as_posix())
    parser.add_argument("--output", default=OUTPUT.as_posix())
    args = parser.parse_args()
    try:
        result = build_view(args.source, args.output)
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print("Staged {} images: {}".format(result["total"], result["output_path"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
