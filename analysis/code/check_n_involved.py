"""Does the 'vehicles and persons involved' column (n_involved) carry the result?

Permutation importance ranks n_involved second, and in about 1% of rows it is larger than the vehicle count
while those rows are mostly fatal. This script refits XGBoost (default and class weights) on 2019-2024 and
scores 2025 with and without that column, and prints how the column relates to the casualty counts.
It changes nothing on the website. The 2026 test year is not used.

Usage:  python -X utf8 check_n_involved.py <combined_csv>
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

import baseline_models as bm
import experiments_thai as ex


def main(csv):
    df = bm.load(csv, "all")
    X = bm.build_features(df)
    y = df["severity"].map({c: i for i, c in enumerate(bm.CLASSES)}).to_numpy()
    year = df["acc_year"].to_numpy()
    tr, va = year <= 2024, year == 2025

    n_inv = pd.to_numeric(df["n_involved"], errors="coerce")
    n_veh = pd.to_numeric(df["n_vehicles"], errors="coerce")
    both = n_inv.notna() & n_veh.notna()
    extra = both & (n_inv > n_veh)
    print(f"rows with both counts: {int(both.sum()):,}; n_involved == n_vehicles: {float((n_inv == n_veh)[both].mean()):.3%}; "
          f"n_involved > n_vehicles: {int(extra.sum()):,} rows, fatal share {float((df.severity[extra] == 'fatal').mean()):.1%} "
          f"(all other rows {float((df.severity[both & ~extra] == 'fatal').mean()):.1%})")
    inj = pd.to_numeric(df["injured_total"], errors="coerce")
    dead = pd.to_numeric(df["deaths"], errors="coerce")
    print("among rows with n_involved > n_vehicles: mean extra persons "
          f"{float((n_inv - n_veh)[extra].mean()):.2f}; mean deaths {float(dead[extra].mean()):.2f}; mean injured {float(inj[extra].mean()):.2f}")
    print("share of all fatal incidents that sit in those rows:", f"{float((df.severity[extra] == 'fatal').sum() / (df.severity == 'fatal').sum()):.1%}")

    y_fatal = (y[va] == 0).astype(int)
    print("\nXGBoost, train 2019-2024, validate 2025")
    print(f"{'variant':14s} {'features':22s} {'fatal F1':>9s} {'fatal PR-AUC':>13s} {'macro-F1':>9s} {'fatal recall':>13s}")
    blanked = X.copy()
    blanked["n_involved"] = np.nan   # the pipeline expects the column, so it is emptied instead of removed
    for variant in ("default", "class_weight"):
        for name, frame in (("all 17", X), ("n_involved emptied", blanked)):
            model = ex.fit_variant("xgboost", variant, frame[tr], y[tr])
            proba = model.predict_proba(frame[va])
            pred = proba.argmax(axis=1)
            print(f"{variant:14s} {name:22s} {f1_score(y[va], pred, labels=[0], average=None)[0]:9.3f} "
                  f"{average_precision_score(y_fatal, proba[:, 0]):13.3f} {f1_score(y[va], pred, average='macro'):9.3f} "
                  f"{float(((pred == 0) & (y[va] == 0)).sum() / (y[va] == 0).sum()):13.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
