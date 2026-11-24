"""Score the 2026 test year ONCE, with the model chosen by aggregate_final.py.

Usage:  python -X utf8 final_test_2026.py <combined_csv> <results_dir>

Reads results/final_selection.json (the pre-set selection rule was applied to the 2022-2025 validation years),
trains that model and variant on 2019-2025, and scores 2026. If the selection is a tuned model, its settings are
tuned first inside 2019-2025 (the last year, 2025, is the inner validation year), so 2026 plays no part in any choice.

The majority-class baseline is scored on the same rows for reference.
Refuses to run if results/test_2026_final.csv already exists: the test year is scored once. Delete that file by
hand only if the run itself failed.

Writes test_2026_final.csv, test_2026_confusion.json and test_2026_bootstrap.csv (95% percentile intervals from
1,000 bootstrap resamples of the 2026 incidents, with the predictions held fixed).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import baseline_models as bm
import cv_tools as cv
from sklearn.metrics import confusion_matrix

N_BOOT = 1000


def main(csv, results_dir):
    results = Path(results_dir)
    out_csv = results / "test_2026_final.csv"
    if out_csv.exists():
        raise SystemExit(f"{out_csv} exists: the 2026 test year has already been scored once. Not running again.")
    sel = json.loads((results / "final_selection.json").read_text(encoding="utf-8"))
    cv.log("selection:", sel)

    df = bm.load(csv, "all")
    df = df[df["acc_year"].notna()].reset_index(drop=True)
    X = bm.build_features(df)
    y = df["severity"].map({c: i for i, c in enumerate(bm.CLASSES)}).to_numpy()
    year = df["acc_year"].to_numpy()
    train, test = year <= 2025, year == 2026
    cat, num = bm.CATEGORICAL, bm.NUMERIC
    cv.log(f"train 2019-2025: {int(train.sum()):,} | test 2026: {int(test.sum()):,}, fatal share {np.mean(y[test] == 0):.4f}")

    params = {}
    if sel["tuned"]:
        params, ap = cv.tune_params(sel["model"], X[train], y[train], year[train], cat, num, 9)
        cv.log("tuned settings:", json.dumps(params, sort_keys=True), f"(inner 2025 PR-AUC {ap:.4f})")

    def predict(name, variant, p):
        if variant == "prior_adjusted":
            proba = cv.fit_variant(name, "default", X[train], y[train], cat, num, p).predict_proba(X[test])
            prior = np.bincount(y[train], minlength=3) / train.sum()
            return proba, (proba / prior).argmax(axis=1)
        proba = cv.fit_variant(name, variant, X[train], y[train], cat, num, p, {}, "final").predict_proba(X[test])
        return proba, proba.argmax(axis=1)

    proba, pred = predict(sel["model"], sel["variant"], params)
    base_proba, base_pred = predict("majority_class", "default", {})
    rows = []
    for label, p_, q_ in ((f"{sel['model']}{'_tuned' if sel['tuned'] else ''}", proba, pred), ("majority_class", base_proba, base_pred)):
        row = {"model": label, "variant": sel["variant"] if label != "majority_class" else "default", "train_years": "2019-2025",
               "test_year": 2026, "n_train": int(train.sum()), "n_test": int(test.sum()),
               "fatal_share_test": float(np.mean(y[test] == 0)), "params": json.dumps(params, sort_keys=True) if label != "majority_class" else "{}"}
        row.update(cv.score(y[test], p_, q_))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    cm = confusion_matrix(y[test], pred, labels=[0, 1, 2])
    (results / "test_2026_confusion.json").write_text(json.dumps({"classes": bm.CLASSES, "rows": "actual", "columns": "predicted",
                                                                   "matrix": cm.tolist()}, indent=2), encoding="utf-8")

    rng = np.random.default_rng(cv.SEED)
    boots = []
    n = int(test.sum())
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[test][idx])) < 3:
            continue
        s = cv.score(y[test][idx], proba[idx], pred[idx])
        boots.append({k: s[k] for k in ("macro_f1", "fatal_recall", "fatal_precision", "fatal_f1", "fatal_pr_auc", "fatal_auc", "balanced_acc")})
    boots = pd.DataFrame(boots)
    ci = pd.DataFrame({"metric": boots.columns, "ci_low": boots.quantile(.025).to_numpy(), "ci_high": boots.quantile(.975).to_numpy()})
    ci.to_csv(results / "test_2026_bootstrap.csv", index=False)

    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).drop(columns=["params"]).round(4).to_string(index=False))
    print(ci.round(4).to_string(index=False))
    print("confusion (rows actual fatal/injury/property_only):", cm.tolist())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
