"""Final experiments on the Thai MOT accident data: SMOTE, nested tuning and the motorcycle-only subset.

Usage:
    python -X utf8 experiments_final.py <combined_csv> <results_dir> --stage smote_untuned|tuned|motorcycle|motorcycle_tuned
                                        [--sample 0.03] [--n-random 9]

Stages (all use the Milestone 3 rolling-origin folds: train 2019-2021 -> 2022 ... 2019-2024 -> 2025;
the 2026 test year is never touched here):
    smote_untuned  the five Milestone 3 model families with SMOTE-NC and their fixed Milestone 3 settings
    tuned          XGBoost and LightGBM tuned inside each fold's training years (nested, time-aware), then each
                   of the four variants fitted with the tuned settings
    motorcycle     the same families and four variants, untuned, on incidents whose first vehicle is a motorcycle
                   (for the comparison with Mohamad et al. 2025)
    motorcycle_tuned  random forest (the benchmark's family), XGBoost and LightGBM tuned inside each fold on the motorcycle
                   subset, with fatal-versus-rest ROC-AUC as the tuning criterion (the benchmark reports AUC). Fixed before
                   running: the same nine random configurations plus the default, strategies default / class weights /
                   prior adjustment, all three families reported (the headline is the random forest).

Set CV_NJOBS to limit the cores used by the tree models (default: all).
"""
import argparse
from pathlib import Path

import numpy as np

import baseline_models as bm
import cv_tools as cv

FOLDS = [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]
FAMILIES = ["logistic_regression", "decision_tree", "random_forest", "xgboost", "lightgbm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("results_dir")
    ap.add_argument("--stage", required=True, choices=["smote_untuned", "tuned", "motorcycle", "motorcycle_tuned"])
    ap.add_argument("--sample", type=float, default=1.0, help="random fraction of rows, for smoke tests only")
    ap.add_argument("--n-random", type=int, default=9, help="random configurations per tuning run (plus the default)")
    args = ap.parse_args()
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = bm.load(args.csv, "motorcycle" if args.stage.startswith("motorcycle") else "all")
    df = df[df["acc_year"].notna()].reset_index(drop=True)   # one row has no parsable date: it cannot be placed in a fold
    if args.sample < 1.0:
        df = df.sample(frac=args.sample, random_state=cv.SEED).reset_index(drop=True)
    X = bm.build_features(df)
    y = df["severity"].map({c: i for i, c in enumerate(bm.CLASSES)}).to_numpy()
    year = df["acc_year"].to_numpy()
    cv.log(f"stage {args.stage}: {len(df):,} incidents, fatal share {np.mean(y == 0):.3f}, years {int(year.min())}-{int(year.max())}")

    cat, num = bm.CATEGORICAL, bm.NUMERIC
    if args.stage == "smote_untuned":
        cv.run_cv(X, y, year, FOLDS, FAMILIES, ["smote"], cat, num, out_dir / "cv_thai_smote_untuned_folds.csv", dataset="thai")
    elif args.stage == "tuned":
        cv.run_cv(X, y, year, FOLDS, ["xgboost", "lightgbm"], cv.VARIANTS, cat, num, out_dir / "cv_thai_tuned_folds.csv",
                  label_suffix="_tuned", tune_names=("xgboost", "lightgbm"), n_random=args.n_random,
                  tuning_path=out_dir / "tuning_thai.csv", dataset="thai")
    elif args.stage == "motorcycle_tuned":
        names = ("random_forest", "xgboost", "lightgbm")
        cv.run_cv(X, y, year, FOLDS, list(names), ["default", "class_weight", "prior_adjusted"], cat, num,
                  out_dir / "cv_motorcycle_tuned_folds.csv", label_suffix="_tuned", tune_names=names, n_random=args.n_random,
                  tuning_path=out_dir / "tuning_motorcycle.csv", dataset="motorcycle", tune_metric="roc_auc")
    else:
        cv.run_cv(X, y, year, FOLDS, ["majority_class"] + FAMILIES, cv.VARIANTS, cat, num, out_dir / "cv_motorcycle_folds.csv",
                  dataset="motorcycle")
    cv.log("done")


if __name__ == "__main__":
    main()
