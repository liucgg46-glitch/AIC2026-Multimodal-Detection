"""E002 weight preflight; delegate all training to the unchanged shared entry."""
import sys

import yaml
import train_rgb


def main():
    train_rgb.configure_console_encoding()
    args = train_rgb.parse_args()
    try:
        config = train_rgb.load_config(train_rgb.project_path(args.config))
        expected = train_rgb.project_path("weights/yolo11n.pt")
        model = train_rgb.project_path(config["model"])
        if model != expected or not model.is_file() or model.stat().st_size == 0:
            raise train_rgb.TrainingConfigError(
                "E002 requires the nonempty local weights/yolo11n.pt from the E001 "
                "initial artifact, with matching SHA256. Automatic download is forbidden."
            )
        if config.get("resume") is not False or config.get("pretrained") is not True:
            raise train_rgb.TrainingConfigError("E002 requires resume=false and pretrained=true")
    except (OSError, train_rgb.TrainingConfigError, yaml.YAMLError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return train_rgb.main()


if __name__ == "__main__":
    raise SystemExit(main())
