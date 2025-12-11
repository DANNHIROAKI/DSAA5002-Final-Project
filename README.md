# DSAA5002-Final-Project

Project scaffold for the customer segmentation coursework. See `customer_segmentation/` for the organized codebase, configs, notebooks, and report assets.


# 默认跑完所有实验
python -m customer_segmentation.run_all_experiments

# 指定随机种子
python -m customer_segmentation.run_all_experiments --seed 2025

# 只跑 RAJC 及其消融（跳过 baselines 和 downstream）
python -m customer_segmentation.run_all_experiments --skip-baselines --skip-downstream
