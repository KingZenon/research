"""Control for the SMOTE variant: is its weaker result caused by resampling, or by the median imputation it needs?

SMOTE-NC cannot take missing numbers, so the SMOTE variant median-imputes the numeric columns (in the training years only),
while the plain models leave the gaps to the trees. This script fits the plain XGBoost and LightGBM (fixed Milestone 3
settings, no resampling, no class weights) on exactly the same imputed matrix as the SMOTE variant and scores the same
four validation years. If these scores match the plain models, imputation is not the reason.

Usage:  python -X utf8 check_smote_control.py <combined_csv> <results_dir>
Writes results/cv_thai_smote_control_folds.csv. The 2026 test year is not used.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

import baseline_models as bm
import cv_tools as cv

FOLDS = [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]


def main(csv, results_dir):
    df = bm.load(csv, "all")
    df = df[df["acc_year"].notna()].reset_index(drop=True)
    X = bm.build_features(df)
    y = df["severity"].map({c: i for i, c in enumerate(bm.CLASSES)}).to_numpy()
    year = df["acc_year"].to_numpy()
    cat, num = bm.CATEGORICAL, bm.NUMERIC
    out = Path(results_dir) / "cv_thai_smote_control_folds.csv"
    out.unlink(missing_ok=True)
    for last_train, val_year in FOLDS:
        tr, va = year <= last_train, year == val_year
        prep = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat),
                                  ("num", SimpleImputer(strategy="median"), num)])
        Ztr, Zva = prep.fit_transform(X[tr]), prep.transform(X[va])
        rows = []
        for name in ("xgboost", "lightgbm"):
            clf = cv.make_model(name, cat, num).named_steps["clf"]
            clf.fit(Ztr, y[tr])
            proba = clf.predict_proba(Zva)
            rows.append(cv._row(f"{name}_imputed_control", "default", val_year, tr, va, y, proba, proba.argmax(axis=1), {}, None, 0.0))
            cv.log(f"val {val_year} {name} imputed, no resampling: fatal F1 {rows[-1]['fatal_f1']:.3f} recall {rows[-1]['fatal_recall']:.3f} "
                   f"PR-AUC {rows[-1]['fatal_pr_auc']:.3f}")
        cv.append_rows(out, rows)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
