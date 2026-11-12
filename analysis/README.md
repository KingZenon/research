# Analysis code and reproduction guide

Code for the ICT324 road accident severity project (Team AHT). It audits the Thai Ministry of Transport (TRAMS) accident files, builds a harmonised table with a three-class severity target (fatal, injury, property damage only), runs the model experiments, and writes the numbers shown on the project website.

Data access and the audit are described in [DATA.md](DATA.md). The raw CSV files are **not** stored in this repository; download them from the publisher (see DATA.md).

## Requirements

Developed and tested with Python 3.14, plus the packages in `code/requirements.txt`:

```
pip install -r code/requirements.txt
```

## Steps

Put the eight downloaded files `accident2019.csv` … `accident2026.csv` in a folder, for example `data/raw/`. Run the scripts from `analysis/code/`. The `-X utf8` flag is needed on Windows because the column names are Thai.

| Step | Command | Output |
| --- | --- | --- |
| 1. Inspect schema, sizes, encodings | `python -X utf8 explore_accidents.py ../../data/raw` | printed summary |
| 2. Harmonise and profile the target | `python -X utf8 harmonize_accidents.py ../../data/raw ../../data/processed/accidents_combined.csv` | merged CSV and printed audit |
| 3. Temporal cross-validation with imbalance variants | `python -X utf8 experiments_thai.py ../../data/processed/accidents_combined.csv ../results` | `results/cv_temporal_thai_folds.csv`, `results/cv_temporal_thai_summary.csv` |
| 4. Website numbers | `python -X utf8 export_site_results.py ../../data/processed/accidents_combined.csv ../results ../..` | `results-data.js` in the site root |

`baseline_models.py` holds the feature builder and model definitions used by step 3. It can also be run on its own for a single split (train 2019–2024, validate 2025).

## Experiment design

- **Target:** fatal (at least one death), injury (no death, at least one injured), property damage only.
- **Features:** province, agency, first vehicle type, road location, presumed cause, accident type, weather, GPS, kilometre marker, vehicles and persons involved, hour, month, weekday, weekend and night flags. Casualty columns define the target and are excluded.
- **Folds (rolling origin):** train 2019–2021 → validate 2022; 2019–2022 → 2023; 2019–2023 → 2024; 2019–2024 → 2025. The 2026 test year is never scored by these scripts.
- **Models:** majority-class baseline, logistic regression, decision tree, random forest, XGBoost, LightGBM, all with fixed, untuned settings (see `baseline_models.py`) and a fixed random seed (42).
- **Variants:** `default` (plain model); `class_weight` (balanced class weights in training); `prior_adjusted` (the plain model's class probabilities divided by the training-set class shares before choosing the label).
- **Reported:** mean and standard deviation over the four folds for fatal recall, precision, F1, PR-AUC and AUC, plus macro-F1, balanced accuracy, MCC and accuracy.

## Results files

- `results/cv_temporal_thai_summary.csv`: one row per model and variant, mean and standard deviation of every metric.
- `results/cv_temporal_thai_folds.csv`: one row per model, variant and fold.
- `../results-data.js`: generated from the two files above; the website's charts and results table read it. Do not edit it by hand.
