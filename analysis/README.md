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

| 5. Confusion matrices, precision-recall points, importance and province map | `python -X utf8 make_site_figures.py ../../data/processed/accidents_combined.csv ../../data/geo/tha_adm1_simplified.geojson ../results ../..` | `results/confusion_matrices_2025.csv`, `results/pr_curves_fatal_2025.csv`, `results/permutation_importance_2025.csv`, `results/province_fatal_share.csv`, and `figures-data.js` and `map-data.js` in the site root |

Step 5 refits the models on 2019–2024 and scores 2025 (the last fold of step 3), and stops with an error if the macro-F1 or fatal F1 differs from `cv_temporal_thai_folds.csv`. It never predicts 2026. If the GeoJSON file is missing it downloads the pinned geoBoundaries release (Thai provinces, ADM1, simplified; OpenStreetMap data under the Open Database License 1.0).

`baseline_models.py` holds the feature builder and model definitions used by step 3. It can also be run on its own for a single split (train 2019–2024, validate 2025).

## Final experiments (Milestone 4)

These steps run after step 3. `cv_tools.py` holds the shared code (SMOTE-NC, nested tuning, rolling-origin folds). Set `CV_NJOBS` to limit the cores used by the tree models. Do not run the scripts with `python -I` (they import each other).

| Step | Command | Output |
| --- | --- | --- |
| 6a. SMOTE-NC, five families, fixed settings | `python -X utf8 experiments_final.py ../../data/processed/accidents_combined.csv ../results --stage smote_untuned` | `results/cv_thai_smote_untuned_folds.csv` |
| 6b. Nested tuning of XGBoost and LightGBM, four variants | `... --stage tuned` | `results/cv_thai_tuned_folds.csv`, `results/tuning_thai.csv` |
| 6c. Motorcycle-only subset (for the comparison with Mohamad et al. 2025) | `... --stage motorcycle` | `results/cv_motorcycle_folds.csv` |
| 6d. Motorcycle subset, tuned (random forest, XGBoost, LightGBM; criterion: fatal-vs-rest ROC-AUC) | `... --stage motorcycle_tuned` | `results/cv_motorcycle_tuned_folds.csv`, `results/tuning_motorcycle.csv` |
| 7. UK STATS19 replication (2021–2024) | `python -X utf8 uk_replication.py ../../data/uk_stats19/raw ../results` | `results/cv_uk_folds.csv`, `results/uk_audit.json` |
| 8. Summaries and the pre-set model selection | `python -X utf8 aggregate_final.py ../results` | `results/cv_*_summary.csv`, `results/selection_report.csv`, `results/final_selection.json` |
| 9. Score the 2026 test year once | `python -X utf8 final_test_2026.py ../../data/processed/accidents_combined.csv ../results` | `results/test_2026_final.csv`, `results/test_2026_confusion.json`, `results/test_2026_bootstrap.csv` |
| 10. Website numbers for the final results | `python -X utf8 export_final_results.py ../../data/processed/accidents_combined.csv ../results ../..` | `final-data.js` in the site root |
| 11. How different the 2023 category spellings are | `python -X utf8 check_2023_categories.py ../../data/processed/accidents_combined.csv ../results` | `results/category_recode_2023.json` |
| 12. Paper figures and numbers | `python -X utf8 make_paper_figures.py ../../data/processed/accidents_combined.csv ../../data/geo/tha_adm1_simplified.geojson ../results ../../paper` | `paper/figures/*.png`, `paper/paper_numbers.json` |
| 13. The paper (.docx) | `python -X utf8 make_paper.py ../results ../../paper` | `paper/Team_AHT_ICT324_Final_Paper.docx` (every number is read from the result files; needs python-docx) |

- **Variants:** the three from Milestone 3 plus `smote`: SMOTE-NC on the training years only, fatal oversampled to the size of the smaller of the other two classes. SMOTE needs complete data, so numeric gaps are median-imputed for this variant.
- **Tuning (nested, time-aware):** for each outer fold the last training year is an inner validation year. The default configuration plus nine random configurations (same list for every fold) are fitted on the earlier years, and the one with the best fatal-class PR-AUC on the inner year wins. All four variants of a model then share the tuned settings. The outer validation year is never seen by tuning. Only XGBoost and LightGBM are tuned, because they led the Milestone 3 cross-validation.
- **Selection rule, fixed before the test year:** among all Thai candidates, the model and variant with the highest mean fatal-class F1 over the four validation years (ties: macro-F1). `selection_report.csv` lists the runners-up with their fold-by-fold gaps, so gaps inside one standard deviation can be called comparable.
- **Test year:** `final_test_2026.py` trains the selected model on 2019–2025 (re-tuning inside those years if it is a tuned model) and scores 2026 once. It refuses to run again once `test_2026_final.csv` exists. The bootstrap file gives 95% intervals from 1,000 resamples of the 2026 incidents.
- **UK replication:** train 2021 → 2022, 2021–2022 → 2023, 2021–2023 → 2024, with the fixed Milestone 3 settings (nothing is tuned on UK data). UK classes are fatal / serious / slight, which are not the Thai classes (the UK has no damage-only records), so the countries are compared on fatal-class metrics. Left out as outcome-related: number of casualties, whether the police attended, the enhanced and adjusted severity fields, and the LSOA code.

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
- `results/confusion_matrices_2025.csv`: counts of actual against predicted class for every model and variant on the 2025 validation year.
- `results/pr_curves_fatal_2025.csv`: fatal-class precision at recall levels 0.02 to 1.00, for the plain and class-weighted models.
- `results/permutation_importance_2025.csv`: drop in fatal-class PR-AUC when each feature is shuffled (XGBoost with class weights, 5 repeats).
- `results/province_fatal_share.csv`: incidents, fatal incidents, fatal share and a 95% Wilson range for each of the 77 provinces.
- `../figures-data.js` and `../map-data.js`: generated by step 5; the website's confusion matrices, precision-recall chart, importance chart and province map read them. Do not edit them by hand. The province outlines are © OpenStreetMap contributors (ODbL 1.0), via geoBoundaries.
