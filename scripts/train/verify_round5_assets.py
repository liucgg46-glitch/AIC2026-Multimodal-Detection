"""Verify the exact external assets used by the Round 5 experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEIMV2_COMMIT = "1d2ca42171570c713e78fc6a766ec5104b7f4724"
ASSETS: Dict[str, Dict[str, str]] = {
    "yolo11x": {
        "path": "weights/yolo11x.pt",
        "sha256": "7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24",
        "source": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt",
    },
    "deimv2_dinov3_s_coco": {
        "path": "weights/deimv2/deimv2_dinov3_s_coco.pth",
        "sha256": "9491ab33b68ecfc0e34043abb3009599ab1e892fb953a1faad12ef4fca5a35c4",
        "source": "https://drive.google.com/file/d/1MDOh8UXD39DNSew6rDzGFp1tAVpSGJdL/view?usp=sharing",
    },
    "dinov3_vitt_distill": {
        "path": "weights/deimv2/vitt_distill.pt",
        "sha256": "2053b865f4e2673fba3f95f7e7e54ad5ee18143885e3ad27eaabb5b3b9919738",
        "source": "https://drive.google.com/file/d/1YMTq_woOLjAcZnHSYNTsNg7f0ahj5LPs/view?usp=sharing",
    },
}


class AssetError(RuntimeError):
    """Raised when a formal external asset is missing or has changed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected: str) -> str:
    if not path.is_file():
        raise AssetError("缺少外部资产: %s" % path)
    actual = sha256(path)
    if actual.lower() != expected.lower():
        raise AssetError("SHA256 不一致: %s: %s != %s" % (path, actual, expected))
    return actual


def verify_git_checkout(path: Path, expected: str = DEIMV2_COMMIT) -> str:
    if not (path / ".git").is_dir():
        raise AssetError("DEIMv2 目录不是 Git checkout: %s" % path)
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(path), capture_output=True, text=True, check=False
    )
    actual = result.stdout.strip()
    if result.returncode or actual.lower() != expected.lower():
        raise AssetError("DEIMv2 commit 不一致: %s != %s" % (actual or "unknown", expected))
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=str(path), capture_output=True, text=True, check=True
    ).stdout.strip()
    if dirty:
        raise AssetError("DEIMv2 checkout 不是 clean；请勿在固定源码内手改: %s" % dirty.splitlines()[0])
    return actual


def verify_assets(project_root: Path, deim_root: Optional[Path] = None) -> Dict[str, object]:
    report: Dict[str, object] = {"assets": {}}
    for name, spec in ASSETS.items():
        path = project_root / spec["path"]
        actual = verify_file(path, spec["sha256"])
        report["assets"][name] = {"path": str(path.resolve()), "sha256": actual, "source": spec["source"]}
    if deim_root is not None:
        report["deimv2"] = {"path": str(deim_root.resolve()), "commit": verify_git_checkout(deim_root)}
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--deim-root", type=Path)
    parser.add_argument("--only", choices=("all", "yolo", "deim"), default="all")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        selected = {
            "all": tuple(ASSETS),
            "yolo": ("yolo11x",),
            "deim": ("deimv2_dinov3_s_coco", "dinov3_vitt_distill"),
        }[args.only]
        report = {"assets": {}}
        for name in selected:
            spec = ASSETS[name]
            path = args.project_root.resolve() / spec["path"]
            report["assets"][name] = {
                "path": str(path.resolve()), "sha256": verify_file(path, spec["sha256"]),
                "source": spec["source"],
            }
        if args.deim_root is not None:
            report["deimv2"] = {
                "path": str(args.deim_root.resolve()),
                "commit": verify_git_checkout(args.deim_root.resolve()),
            }
    except (AssetError, OSError, subprocess.SubprocessError) as exc:
        print("错误: %s" % exc)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
