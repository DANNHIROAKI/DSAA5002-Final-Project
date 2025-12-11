"""Script placeholder to train and evaluate the RAJC model."""
from pathlib import Path
import yaml


def load_config(config_path: Path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config(Path("customer_segmentation/configs/rajc.yaml"))
    # TODO: implement RAJC training and evaluation pipeline
    print("Loaded RAJC config:", config.get("rajc"))


if __name__ == "__main__":
    main()
