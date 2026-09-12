"""E003 artifact and dataset preflight; reuse the unchanged training entry."""
import hashlib
import sys

import yaml
import train_rgb


INITIAL_WEIGHT_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
DATASET = "data/processed/depth_trainable/inverse/data.yaml"


def main():
    train_rgb.configure_console_encoding()
    args = train_rgb.parse_args()
    try:
        config = train_rgb.load_config(train_rgb.project_path(args.config))
        if train_rgb.project_path(config["data"]) != train_rgb.project_path(DATASET):
            raise train_rgb.TrainingConfigError("E003 requires the canonical Depth inverse data.yaml")
        model = train_rgb.project_path(config["model"])
        if model != train_rgb.project_path("weights/yolo11n.pt") or not model.is_file():
            raise train_rgb.TrainingConfigError(
                "E003 requires local weights/yolo11n.pt; automatic download is forbidden"
            )
        digest = hashlib.sha256()
        with model.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != INITIAL_WEIGHT_SHA256:
            raise train_rgb.TrainingConfigError("E003 initial weight SHA256 mismatch")
        if config.get("resume") is not False or config.get("pretrained") is not True:
            raise train_rgb.TrainingConfigError("E003 requires resume=false and pretrained=true")
    except (OSError, train_rgb.TrainingConfigError, yaml.YAMLError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return train_rgb.main()


if __name__ == "__main__":
    raise SystemExit(main())
