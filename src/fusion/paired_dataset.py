"""Read-only canonical pairing; geometry operates on one six-channel container."""
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset
from ultralytics.data.augment import Compose, Format, LetterBox, RandomFlip, RandomHSV
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.instance import Instances

from scripts.data import prepare_ir_yolo as canonical
from . import require_ultralytics

ROOT = Path(__file__).resolve().parents[2]
DISABLED = ("degrees", "translate", "scale", "shear", "perspective", "mosaic", "mixup", "cutmix", "copy_paste", "multi_scale", "bgr")


def check_augmentation(hyp):
    for key in DISABLED:
        if getattr(hyp, key, 0):
            raise ValueError(f"Fusion v1 disables augmentation: {key}")
    if getattr(hyp, "augmentations", None):
        raise ValueError("Fusion v1 disables custom/Albumentations augmentations")


def read_raw3(path):
    # Surface OS access errors instead of misreporting them as corrupt image encoding.
    with Path(path).open("rb"):
        pass
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected uint8 raw3 image: {path}")
    return image


def pair_records(rgb_dir, ir_dir, label_dir, splits, counts=(1600, 400), ir_label_dir=None):
    """Strict low-level pairing; optional view labels must byte-match labels_clean."""
    groups = {mode: canonical.read_stems(Path(splits[mode]), count)
              for mode, count in zip(("train", "val"), counts)}
    all_stems = groups["train"] + groups["val"]
    if len({s.casefold() for s in all_stems}) != len(all_stems):
        raise ValueError("train/val split overlap")
    rgb = canonical.index_ir_images(Path(rgb_dir))
    ir = canonical.index_ir_images(Path(ir_dir))
    labels = canonical.index_labels(Path(label_dir))
    for name, mapping in (("RGB", rgb), ("IR", ir), ("labels", labels)):
        if set(mapping) != set(all_stems):
            raise ValueError(f"{name} missing/extra stem or split mismatch")
    if ir_label_dir is not None:
        ir_labels = canonical.index_labels(Path(ir_label_dir))
        if set(ir_labels) != set(labels):
            raise ValueError("IR label split mismatch")
        for stem in all_stems:
            if ir_labels[stem].read_bytes() != labels[stem].read_bytes():
                raise ValueError(f"label mismatch: {stem}")
    return {mode: [(s, rgb[s], ir[s], labels[s]) for s in stems] for mode, stems in groups.items()}


def canonical_records():
    labels = ROOT / canonical.CANONICAL_LABEL_RELATIVE
    splits = {mode: ROOT / f"data/splits/{mode}.txt" for mode in ("train", "val")}
    canonical.validate_canonical_inputs(labels, splits["train"], splits["val"])
    records = pair_records(ROOT / "data/raw/train/visible", ROOT / canonical.CANONICAL_IR_RELATIVE, labels, splits)
    identity, _ = canonical._aggregate_paths(list(canonical.index_labels(labels).values()))
    if identity["aggregate_sha256"] != canonical.CANONICAL_LABELS_CLEAN_SHA256:
        raise ValueError("canonical labels_clean identity mismatch")
    return records


class RGBOnlyHSV:
    def __init__(self, hyp):
        self.hsv = RandomHSV(hyp.hsv_h, hyp.hsv_s, hyp.hsv_v)

    def __call__(self, labels):
        # Both triplets are BGR here. Never pass IR to HSV.
        rgb = {"img": np.ascontiguousarray(labels["img"][..., :3])}
        self.hsv(rgb)
        labels["img"][..., :3] = rgb["img"]
        return labels


class PairedFormat(Format):
    def __call__(self, labels):
        # Convert each BGR triplet separately; reversing all six would swap modalities.
        labels["img"] = np.ascontiguousarray(labels["img"][..., [2, 1, 0, 5, 4, 3]])
        return super().__call__(labels)


class PairedLetterBox(LetterBox):
    def __call__(self, labels):
        # 8.3.253 only appends pad to ratio_pad; 8.4.144 also multiplies resize gain.
        # Our pipeline has no preceding resize. Normalize metadata for BOTH versions.
        h, w = labels["img"].shape[:2]
        target_shape = labels.get("rect_shape", self.new_shape)
        gain = min(target_shape[0] / h, target_shape[1] / w)
        if not self.scaleup:
            gain = min(gain, 1.0)
        result = super().__call__(labels)
        result["ratio_pad"] = ((gain, gain), result["ratio_pad"][1])
        return result


class PairedDataset(Dataset):
    collate_fn = staticmethod(YOLODataset.collate_fn)
    rect = False
    mosaic = False

    def __init__(self, records, imgsz, hyp, augment=False):
        require_ultralytics()
        check_augmentation(hyp)
        if imgsz < 32 or imgsz % 32:
            raise ValueError("imgsz must be a positive multiple of 32")
        self.records, self.imgsz, self.augment = records, imgsz, augment
        self.im_files = [str(r[1]) for r in records]
        self.labels = []
        for stem, rgb_path, ir_path, label_path in records:
            # E002 raw3 is a byte-preserving copy/link of this exact raw source.
            rgb, ir = read_raw3(rgb_path), read_raw3(ir_path)
            if rgb.shape != ir.shape:
                raise ValueError(f"paired shape mismatch: {stem}")
            rows = [line.split() for line in label_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            if any(len(row) != 5 for row in rows):
                raise ValueError(f"invalid label columns: {stem}")
            a = np.asarray(rows, dtype=np.float32).reshape(-1, 5)
            if (not np.isfinite(a).all() or (a[:, 0] != a[:, 0].astype(int)).any()
                    or (a[:, 0] < 0).any() or (a[:, 0] > 11).any()
                    or (a[:, 1:] < 0).any() or (a[:, 1:] > 1).any() or (a[:, 3:] <= 0).any()):
                raise ValueError(f"invalid labels: {stem}")
            self.labels.append(dict(im_file=str(rgb_path), shape=rgb.shape[:2], cls=a[:, :1], bboxes=a[:, 1:]))
        transforms = [PairedLetterBox((imgsz, imgsz), scaleup=augment)]
        if augment:
            transforms += [RandomFlip(hyp.flipud, "vertical"), RandomFlip(hyp.fliplr, "horizontal"), RGBOnlyHSV(hyp)]
        transforms.append(PairedFormat(bbox_format="xywh", normalize=True, batch_idx=True))
        self.transforms = Compose(transforms)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        stem, rgb_path, ir_path, _ = self.records[index]
        rgb, ir = read_raw3(rgb_path), read_raw3(ir_path)
        if rgb.shape != ir.shape:
            raise ValueError(f"paired shape mismatch: {stem}")
        label = deepcopy(self.labels[index])
        label.pop("shape")
        boxes = label.pop("bboxes")
        label.update(img=np.concatenate((rgb, ir), axis=2), ori_shape=rgb.shape[:2],
                     ratio_pad=(1.0, 1.0), instances=Instances(boxes, np.zeros((0, 1000, 2), dtype=np.float32),
                                                            bbox_format="xywh", normalized=True))
        return self.transforms(label)
