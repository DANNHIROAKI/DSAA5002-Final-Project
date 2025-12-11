# DSAA5002 – Response-Aware Customer Segmentation

This package contains the code for a DSAA5002 final project on **response‑aware
customer segmentation** based on the Kaggle *Customer Personality Analysis*
dataset. It implements:

- Several clustering **baselines**:
  - RFM K‑Means
  - Full‑feature K‑Means
  - Gaussian Mixture Model (GMM)
  - Cluster‑then‑Predict (cluster first, then fit per‑cluster classifiers)
- A proposed **RAJC** model (Response‑Aware Joint Clustering) that jointly
  optimizes customer similarity and promotion‑response behavior.
- An **end‑to‑end launcher** to reproduce all experiments and figures used in
  the report.

> The raw Kaggle CSV (`marketing_campaign.csv`) is **not tracked in git**.
> You must place it under `customer_segmentation/data/raw/` before running any
> pipeline.

---

## Project Layout

High‑level structure of the `customer_segmentation` package:

```text
customer_segmentation/
│
├─ README.md                 # This file
├─ run_all_experiments.py    # One‑shot launcher for all experiments + plots
│
├─ data/
│   ├─ raw/                  # Raw Kaggle CSV (marketing_campaign.csv)
│   └─ processed/            # Optional cached features / splits
│
├─ configs/
│   ├─ baselines.yaml        # Hyper‑parameters for all baselines
│   └─ rajc.yaml             # Hyper‑parameters for the RAJC model
│
├─ src/
│   ├─ data/                 # Loading, cleaning, feature engineering
│   ├─ models/               # Baselines + RAJC implementation
│   ├─ evaluation/           # Clustering / segmentation / prediction metrics
│   ├─ visualization/        # Elbow, PCA/t‑SNE, profile plots
│   ├─ experiments/          # Individual experiment entrypoints
│   └─ utils/                # Logging, seeding, small metric helpers
│
└─ outputs/
    ├─ logs/                 # Run logs (including run_all.log)
    ├─ figures/              # All figures used in the report / slides
    ├─ tables/               # CSV tables with metrics
    └─ models/               # (optional) trained model artefacts
