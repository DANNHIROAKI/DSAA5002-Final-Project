# Customer Segmentation Project Structure

This repository organizes the DSAA5002 final project for response-aware customer segmentation. It provides a modular layout for data handling, modeling, evaluation, and reporting.

## Layout
- `data/`: raw and processed datasets.
- `configs/`: YAML configuration files for baselines and the proposed RAJC model.
- `src/`: Python source code grouped by functionality (data, models, evaluation, visualization, experiments, utils).
- `notebooks/`: exploratory analysis and prototyping notebooks.
- `outputs/`: saved logs, figures, tables, and trained models.
- `report/`: manuscript and figures used in the final write-up.

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`.
2. Place the raw dataset at `data/raw/marketing_campaign.csv`.
3. Use scripts under `src/experiments/` to run baselines, RAJC training, and analyses once implemented.

Each Python module currently contains skeleton implementations and docstrings to guide further development.
