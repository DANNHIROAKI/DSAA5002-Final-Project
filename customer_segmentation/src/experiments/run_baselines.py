"""Script placeholder to run baseline clustering methods."""
from pathlib import Path
import yaml


def load_config(config_path: Path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    """Load configs and execute baseline pipelines (stub)."""
    configs = load_config(Path("customer_segmentation/configs/baselines.yaml"))
    # TODO: implement baseline experiment orchestration
    print("Loaded baseline configs:", configs.keys())


if __name__ == "__main__":
    main()
