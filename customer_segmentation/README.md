# Customer Segmentation Project Structure

This repository organizes the DSAA5002 final project for response-aware customer segmentation. It provides a modular layout for data handling, modeling, evaluation, and reporting.

> Note: The raw Kaggle CSV (`marketing_campaign.csv`) is **not tracked in git**. Place it under `data/raw/` before running any pipeline.

## Layout
- `data/`: raw and processed datasets.
- `configs/`: YAML configuration files for baselines and the proposed RAJC model.
- `src/`: Python source code grouped by functionality (data, models, evaluation, visualization, experiments, utils).
- `notebooks/`: exploratory analysis and prototyping notebooks.
- `outputs/`: saved logs, figures, tables, and trained models.
- `report/`: manuscript and figures used in the final write-up.

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`.
2. Place the raw dataset at `data/raw/marketing_campaign.csv`. Run `python -m customer_segmentation.src.data.check_data` to verify presence.

## Running experiments
All entrypoints assume the working directory is the repository root:

- Baselines (RFM/Full K-Means, GMM, Cluster-then-Predict):
  ```bash
  python -m customer_segmentation.src.experiments.run_baselines
  ```
- Proposed RAJC model:
  ```bash
  python -m customer_segmentation.src.experiments.run_rajc
  ```
- Ablation over λ and cluster counts:
  ```bash
  python -m customer_segmentation.src.experiments.run_ablation
  ```
- Downstream promotion-response prediction with cluster IDs:
  ```bash
  python -m customer_segmentation.src.experiments.run_downstream
  ```

Each script will report a clear error if the dataset file is missing and saves tables under `customer_segmentation/outputs/tables/`.
