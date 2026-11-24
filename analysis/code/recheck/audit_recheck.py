"""Data and pipeline audit for the re-check (no model is fitted here).

Usage:  python -X utf8 audit_recheck.py <combined_csv> <results_recheck_dir>

Writes (all under the results folder): audit_missing_by_year.csv, audit_unseen_categories_by_fold.csv,
audit_2023_label_examples.csv, audit_summary.json.  2026 rows are dropped as soon as the CSV is read.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import recheck_common as rc


def main(csv_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df, X, y, year = rc.load_pre2026(csv_path)
    summ = {"rows_2019_2025_after_dedup": int(len(df))}
    rc.log("rows 2019-2025 after the original duplicate rule:", len(df))
    assert len(df) == 151314, "row set differs from the original study's 2019-2025 rows"

    # class shares
    summ["class_share_overall"] = {c: round(float((y == i).mean()), 4) for i, c in enumerate(rc.bm.CLASSES)}
    tab = pd.crosstab(year, df["severity"])
    tab["fatal_share"] = (tab["fatal"] / tab.sum(axis=1)).round(4)
    print(tab)
    summ["fatal_share_by_year"] = tab["fatal_share"].to_dict()

    # missingness of the model inputs by year (after the original normalisation)
    raw_cols = ["time", "province", "agency", "vehicle1", "location", "cause", "acc_type", "weather", "lat", "lon", "km",
                "n_vehicles", "n_involved", "route", "report_date", "report_time", "acc_code"]
    miss = df[raw_cols].isna().groupby(year).mean().round(4).T
    feat_miss = pd.DataFrame({
        "hour_unparsed": X["hour"].isna().groupby(year).mean().round(4),
        "lat_lon_missing_or_outside_Thailand": X["lat"].isna().groupby(year).mean().round(4)}).T
    cat_missing = pd.DataFrame({c: (X[c] == "missing").groupby(year).mean().round(4) for c in rc.CAT}).T
    cat_missing.index = [f"{i}=='missing'" for i in cat_missing.index]
    pd.concat([miss, feat_miss, cat_missing]).to_csv(out / "audit_missing_by_year.csv")

    # duplicates
    nokey = df.drop(columns=["source_file", "acc_code"])
    summ["exact_duplicate_rows_ignoring_source_and_code"] = int(nokey.duplicated().sum())
    k2 = ["date_parsed", "time", "province", "km", "lat", "lon", "vehicle1", "deaths", "injured_total"]
    summ["duplicates_on_date_time_place_vehicle_casualties"] = int(df.duplicated(subset=k2).sum())
    summ["duplicates_on_that_key_by_year"] = df[df.duplicated(subset=k2)].groupby("acc_year").size().astype(int).to_dict()

    # impossible / outlying values
    summ["lat_lon_outside_Thailand_box_or_missing"] = int(X["lat"].isna().sum())
    summ["km_min_max"] = [float(X["km"].min()), float(X["km"].max())]
    summ["km_negative"] = int((X["km"] < 0).sum())
    summ["n_vehicles_quantiles_50_99_999_max"] = [float(X["n_vehicles"].quantile(q)) for q in (.5, .99, .999)] + [float(X["n_vehicles"].max())]
    summ["n_involved_quantiles_50_99_999_max"] = [float(X["n_involved"].quantile(q)) for q in (.5, .99, .999)] + [float(X["n_involved"].max())]
    summ["n_vehicles_zero_or_missing"] = int((X["n_vehicles"].fillna(0) <= 0).sum())
    summ["hour_unparsed_total"] = int(X["hour"].isna().sum())

    # leakage audit of n_involved ("vehicles and persons involved" may count the casualties themselves)
    nv, ni = X["n_vehicles"], X["n_involved"]
    both = nv.notna() & ni.notna()
    extra = (ni - nv)[both]
    casual = (pd.to_numeric(df["deaths"]) + pd.to_numeric(df["injured_total"]))[both]
    persons = (pd.to_numeric(df["deaths"]) + pd.to_numeric(df["injured_total"]))
    summ["n_involved_audit"] = {
        "rows_with_both_counts": int(both.sum()),
        "share_n_involved_equals_n_vehicles": round(float((extra == 0).mean()), 4),
        "share_n_involved_greater": round(float((extra > 0).mean()), 4),
        "share_n_involved_smaller_than_vehicles": round(float((extra < 0).mean()), 4),
        "fatal_share_when_greater": round(float((df.severity[both][extra > 0] == "fatal").mean()), 4),
        "fatal_share_when_equal": round(float((df.severity[both][extra == 0] == "fatal").mean()), 4),
        "share_of_all_fatal_in_rows_with_extra_persons": round(float(((df.severity[both] == "fatal") & (extra > 0)).sum() / (df.severity[both] == "fatal").sum()), 4),
        "corr_extra_persons_with_deaths_plus_injured": round(float(np.corrcoef(extra, casual)[0, 1]), 4),
        "single_feature_fatal_ROC_AUC_extra_persons": round(float(roc_auc_score((df.severity[both] == "fatal").astype(int), extra)), 4),
        "single_feature_fatal_ROC_AUC_n_involved": round(float(roc_auc_score((df.severity[both] == "fatal").astype(int), ni[both])), 4),
        "single_feature_fatal_ROC_AUC_n_vehicles": round(float(roc_auc_score((df.severity[both] == "fatal").astype(int), nv[both])), 4),
    }

    # report lag (post-outcome administrative field) - the original code does not use it, check why it should stay out
    d1 = pd.to_datetime(df["date_parsed"], errors="coerce")
    rd = pd.to_datetime(df["report_date"], dayfirst=True, errors="coerce")
    lag = (rd - d1).dt.days
    ok = lag.notna() & (lag >= 0) & (lag < 400)
    summ["report_date_lag_days_AUC_for_fatal_(not_used_as_feature)"] = round(float(roc_auc_score((df.severity[ok] == "fatal").astype(int), lag[ok])), 4)

    # category audit: cardinality, share of validation rows with labels unseen in the training years, per fold
    rows = []
    for last_train, val_year in rc.FOLDS:
        tr, va = year <= last_train, year == val_year
        sh = rc.unseen_share(X[tr], X[va])
        rows.append({"val_year": val_year, **{c: round(v, 4) for c, v in sh.items()}})
    pd.DataFrame(rows).to_csv(out / "audit_unseen_categories_by_fold.csv", index=False)
    summ["unseen_label_share_by_fold"] = {r["val_year"]: {c: r[c] for c in rc.CAT} for r in rows}
    summ["n_labels_by_column_by_year"] = {c: {int(yr): int(X.loc[year == yr, c].nunique()) for yr in range(2019, 2026)} for c in rc.CAT}

    # examples of 2023 labels versus 2022 labels for the two worst columns
    ex = []
    for c in ("cause", "location", "vehicle1", "weather"):
        for yr in (2022, 2023, 2024):
            vc = X.loc[year == yr, c].value_counts().head(6)
            for lab, n in vc.items():
                ex.append({"column": c, "year": yr, "label": lab, "n": int(n), "share": round(n / (year == yr).sum(), 4)})
    pd.DataFrame(ex).to_csv(out / "audit_2023_label_examples.csv", index=False, encoding="utf-8-sig")

    # features not obviously known at first report
    summ["note_features"] = ("cause = presumed cause (investigator's judgement), acc_type = collision type, n_involved = vehicles AND "
                             "persons involved: all are written on the report after the event; see report text")
    (out / "audit_summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(summ, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
