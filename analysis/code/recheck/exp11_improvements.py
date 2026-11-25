"""Experiment 11 (post hoc, 2022-2025 validation folds only; 2026 is never loaded):
Tests targeted, zero-leakage improvements for:
  1. 2023 Thai category synonym harmonisation (cause & location spellings)
  2. Domain feature engineering from existing TRAMS columns:
     - Thai holiday peak windows (Songkran Apr 11-17, New Year Dec 29-Jan 4)
     - Cyclical hour & month (sin/cos)
     - Explicit extra_persons = max(0, n_involved - n_vehicles) and flag
     - Coarse road_tier extracted from route text (highway number length / agency)
  3. Operating tiers on calibrated plain probabilities (top-5% high-precision tier,
     inner-year F1 threshold, and 5:1 cost threshold)
  4. Dedicated binary fatal-vs-rest motorcycle models (native LightGBM, ordinal/native XGBoost,
     and nested ROC-AUC tuned LightGBM) with harmonised + engineered features, compared on
     the exact 2022-2025 rolling folds against the Mohamad et al. (2025) benchmark (0.768 AUC).

Usage:
    python -X utf8 exp11_improvements.py <combined_csv> <out_json>
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

import recheck_common as rc

CAUSE_MAP = {
    "ขับรถเร็วเกินอัตราที่กำหนด": "ขับรถเร็วเกินอัตรากำหนด",
    "มีการตัดหน้าระยะกระชั้นชิด": "คน/รถ/สัตว์ตัดหน้ากระชั้นชิด",
    "ยางรถยนต์ชำรุด": "ยางเสื่อมสภาพ/ยางแตก",
    "ถนนลื่น(เนื่องจากสภาพอากาศ)": "ถนนลื่น",
}
LOCATION_MAP = {
    "ทางตรง+ที่ราบ": "ทางตรง+ไม่มีความลาดชัน",
    "ทางตรง+ไม่มีความลาดชัน/ที่ราบ": "ทางตรง+ไม่มีความลาดชัน",
    "ทางโค้งปกติ+ที่ราบ": "ทางโค้งกว้าง+ไม่มีความลาดชัน",
    "ทางโค้งปกติ+บนช่วงลาดชัน": "ทางโค้งกว้าง+ที่ลาดชัน",
}


def extract_road_tier(route_series, agency_series):
    """Coarse road class from route text + agency (deterministic, handles missing route)."""
    tiers = []
    for route, agency in zip(route_series.astype("string"), agency_series.astype("string")):
        ag = "" if pd.isna(agency) else str(agency).replace(" ", "")
        if "ทางพิเศษ" in ag:
            tiers.append("expressway")
            continue
        if pd.isna(route) or not str(route).strip():
            tiers.append("missing_route")
            continue
        s = str(route)
        if "มอเตอร์เวย์" in s or "ทางพิเศษ" in s or "ทางหลวงพิเศษ" in s:
            tiers.append("expressway")
            continue
        m = re.search(r"หมายเลข\s*(\d{1,4})", s)
        if m:
            n = len(m.group(1))
            tiers.append("hwy_1_2digit" if n <= 2 else "hwy_3_4digit")
        elif "ชนบท" in ag:
            tiers.append("rural_road")
        else:
            tiers.append("named_route")
    return pd.Series(tiers, index=route_series.index, dtype="string")


def build_upgraded_features(df, X_base, harmonize_2023=True, add_engineered=True):
    X = X_base.copy()
    cats = list(rc.CAT)
    if harmonize_2023:
        X["cause"] = X["cause"].replace(CAUSE_MAP)
        X["location"] = X["location"].replace(LOCATION_MAP)
    if add_engineered:
        when = pd.to_datetime(df["date_parsed"])
        m, d = when.dt.month, when.dt.day
        is_songkran = ((m == 4) & (d >= 11) & (d <= 17)).astype(float)
        is_newyear = (((m == 12) & (d >= 29)) | ((m == 1) & (d <= 4))).astype(float)
        X["is_holiday_peak"] = ((is_songkran > 0) | (is_newyear > 0)).astype(float)
        X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24.0)
        X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24.0)
        X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12.0)
        X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12.0)
        diff = (X["n_involved"] - X["n_vehicles"]).clip(lower=0)
        X["extra_persons"] = diff
        X["has_extra_persons"] = (diff > 0).astype(float)
        X.loc[diff.isna(), "has_extra_persons"] = np.nan
        X["road_tier"] = extract_road_tier(df["route"], df["agency"])
        cats = cats + ["road_tier"]
    return X, cats


def eval_multiclass_lgbm(X, y, year, cats, label):
    fold_rows = []
    tier_rows = {k: [] for k in ("top5", "top10", "thr_f1_inner", "thr_cost5")}
    for last_train, val_year in rc.FOLDS:
        tr, va = year <= last_train, year == val_year
        inner_tr, inner_va = year <= last_train - 1, year == last_train

        # Inner threshold on last training year
        icats = rc.fit_categories(X[inner_tr], cats)
        iclf = rc.make_clf("lightgbm", True, "default")
        iclf.fit(rc.to_native(X[inner_tr], icats, cats), y[inner_tr])
        p_in = iclf.predict_proba(rc.to_native(X[inner_va], icats, cats))[:, 0]
        t_f1, _ = rc.best_f1_threshold((y[inner_va] == 0).astype(int), p_in)
        t_top5 = float(np.quantile(p_in, 0.95))
        t_top10 = float(np.quantile(p_in, 0.90))
        t_cost5 = 1.0 / 6.0

        # Outer fit
        ocats = rc.fit_categories(X[tr], cats)
        oclf = rc.make_clf("lightgbm", True, "default")
        oclf.fit(rc.to_native(X[tr], ocats, cats), y[tr])
        proba = oclf.predict_proba(rc.to_native(X[va], ocats, cats))
        truth = (y[va] == 0).astype(int)
        pf = proba[:, 0]

        pred_f1 = rc.predict_with_threshold(proba, t_f1)
        s = rc.full_score(y[va], proba, pred_f1)
        s["val_year"] = val_year
        s["threshold"] = t_f1
        fold_rows.append(s)

        for name, thr in (("top5", t_top5), ("top10", t_top10), ("thr_f1_inner", t_f1), ("thr_cost5", t_cost5)):
            flag = pf >= thr
            tier_rows[name].append({
                "val_year": val_year,
                "flagged_share": float(flag.mean()),
                "recall": float(recall_score(truth, flag, zero_division=0)),
                "precision": float(precision_score(truth, flag, zero_division=0)),
                "f1": float(f1_score(truth, flag, zero_division=0)),
                "threshold": round(thr, 4),
            })
    fdf = pd.DataFrame(fold_rows)
    res = {
        "label": label,
        "fatal_pr_auc_mean": round(float(fdf["fatal_pr_auc"].mean()), 5),
        "fatal_pr_auc_sd": round(float(fdf["fatal_pr_auc"].std(ddof=1)), 5),
        "fatal_auc_mean": round(float(fdf["fatal_auc"].mean()), 5),
        "fatal_f1_thr_mean": round(float(fdf["fatal_f1"].mean()), 5),
        "fatal_f1_thr_sd": round(float(fdf["fatal_f1"].std(ddof=1)), 5),
        "fatal_recall_thr_mean": round(float(fdf["fatal_recall"].mean()), 5),
        "fatal_precision_thr_mean": round(float(fdf["fatal_precision"].mean()), 5),
        "ece_fatal_mean": round(float(fdf["ece_fatal"].mean()), 5),
        "by_year_pr_auc": {int(r.val_year): round(float(r.fatal_pr_auc), 4) for r in fdf.itertuples()},
        "by_year_f1": {int(r.val_year): round(float(r.fatal_f1), 4) for r in fdf.itertuples()},
        "tiers": {
            name: {
                "flagged_share": round(float(np.mean([x["flagged_share"] for x in items])), 4),
                "recall": round(float(np.mean([x["recall"] for x in items])), 4),
                "precision": round(float(np.mean([x["precision"] for x in items])), 4),
                "f1": round(float(np.mean([x["f1"] for x in items])), 4),
                "threshold": round(float(np.mean([x["threshold"] for x in items])), 4),
            }
            for name, items in tier_rows.items()
        },
    }
    rc.log(f"[MAIN] {label:28s} PR-AUC {res['fatal_pr_auc_mean']:.4f}±{res['fatal_pr_auc_sd']:.4f} "
           f"F1(thr) {res['fatal_f1_thr_mean']:.4f}±{res['fatal_f1_thr_sd']:.4f} "
           f"rec {res['fatal_recall_thr_mean']:.3f} prec {res['fatal_precision_thr_mean']:.3f} ECE {res['ece_fatal_mean']:.4f}")
    return res


def eval_motorcycle_binary(df, X_base, y_all, year_all):
    mask = df["vehicle1"].astype("string").str.contains("จักรยานยนต์", na=False).to_numpy()
    df_m = df[mask].reset_index(drop=True)
    Xb_m = X_base[mask].reset_index(drop=True)
    y_m = (y_all[mask] == 0).astype(int)
    yr_m = year_all[mask]
    rc.log(f"[MOTO] pre-2026 rows={len(df_m):,}, fatal share={y_m.mean():.4f}")

    configs = [
        ("binary_lgbm_native_base", False, False, False),
        ("binary_lgbm_native_upgraded", True, True, False),
        ("binary_lgbm_native_upgraded_tuned", True, True, True),
    ]
    space = [
        {},
        {"num_leaves": 31, "learning_rate": 0.03, "n_estimators": 500, "min_child_samples": 40, "subsample": 0.85, "subsample_freq": 1, "colsample_bytree": 0.7, "reg_lambda": 10.0},
        {"num_leaves": 31, "learning_rate": 0.03, "n_estimators": 400, "min_child_samples": 60, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.6, "reg_lambda": 20.0},
        {"num_leaves": 15, "learning_rate": 0.05, "n_estimators": 500, "min_child_samples": 30, "subsample": 0.85, "subsample_freq": 1, "colsample_bytree": 0.7, "reg_lambda": 5.0},
        {"num_leaves": 63, "learning_rate": 0.03, "n_estimators": 400, "min_child_samples": 50, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.7, "reg_lambda": 15.0},
        {"num_leaves": 23, "learning_rate": 0.03, "n_estimators": 600, "min_child_samples": 50, "subsample": 0.85, "subsample_freq": 1, "colsample_bytree": 0.6, "reg_lambda": 20.0},
        {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 300, "min_child_samples": 80, "subsample": 0.75, "subsample_freq": 1, "colsample_bytree": 0.6, "reg_lambda": 10.0},
        {"num_leaves": 47, "learning_rate": 0.03, "n_estimators": 450, "min_child_samples": 60, "subsample": 0.85, "subsample_freq": 1, "colsample_bytree": 0.7, "reg_lambda": 25.0},
    ]

    results = []
    for name, harm, eng, do_tune in configs:
        Xm, cats = build_upgraded_features(df_m, Xb_m, harmonize_2023=harm, add_engineered=eng)
        rows = []
        for last_train, val_year in rc.FOLDS:
            tr, va = yr_m <= last_train, yr_m == val_year
            best_p = {}
            if do_tune:
                itr, iva = yr_m <= last_train - 1, yr_m == last_train
                icats = rc.fit_categories(Xm[itr], cats)
                best_auc = -1.0
                for cfg in space:
                    kw = dict(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=63,
                              n_jobs=rc.N_JOBS, random_state=rc.SEED, verbose=-1)
                    kw.update(cfg)
                    m = LGBMClassifier(**kw)
                    m.fit(rc.to_native(Xm[itr], icats, cats), y_m[itr])
                    auc_in = float(roc_auc_score(y_m[iva], m.predict_proba(rc.to_native(Xm[iva], icats, cats))[:, 1]))
                    if auc_in > best_auc:
                        best_auc, best_p = auc_in, cfg
            ocats = rc.fit_categories(Xm[tr], cats)
            kw = dict(objective="binary", n_estimators=400, learning_rate=0.05, num_leaves=63,
                      n_jobs=rc.N_JOBS, random_state=rc.SEED, verbose=-1)
            kw.update(best_p)
            clf = LGBMClassifier(**kw)
            clf.fit(rc.to_native(Xm[tr], ocats, cats), y_m[tr])
            p = clf.predict_proba(rc.to_native(Xm[va], ocats, cats))[:, 1]
            pred = (p >= 0.5).astype(int)
            rows.append({
                "val_year": val_year,
                "fatal_auc": float(roc_auc_score(y_m[va], p)),
                "fatal_pr_auc": float(average_precision_score(y_m[va], p)),
                "accuracy": float(accuracy_score(y_m[va], pred)),
                "fatal_recall": float(recall_score(y_m[va], pred, zero_division=0)),
                "fatal_precision": float(precision_score(y_m[va], pred, zero_division=0)),
                "fatal_f1": float(f1_score(y_m[va], pred, zero_division=0)),
                "ece": rc.ece_binary(y_m[va], p),
            })
        rdf = pd.DataFrame(rows)
        entry = {
            "label": name,
            "fatal_auc_mean": round(float(rdf["fatal_auc"].mean()), 5),
            "fatal_auc_sd": round(float(rdf["fatal_auc"].std(ddof=1)), 5),
            "fatal_pr_auc_mean": round(float(rdf["fatal_pr_auc"].mean()), 5),
            "accuracy_mean": round(float(rdf["accuracy"].mean()), 5),
            "fatal_recall_mean": round(float(rdf["fatal_recall"].mean()), 5),
            "fatal_precision_mean": round(float(rdf["fatal_precision"].mean()), 5),
            "fatal_f1_mean": round(float(rdf["fatal_f1"].mean()), 5),
            "ece_mean": round(float(rdf["ece"].mean()), 5),
            "by_year_auc": {int(r.val_year): round(float(r.fatal_auc), 4) for r in rdf.itertuples()},
        }
        results.append(entry)
        rc.log(f"[MOTO] {name:34s} AUC {entry['fatal_auc_mean']:.4f}±{entry['fatal_auc_sd']:.4f} "
               f"acc {entry['accuracy_mean']:.4f} rec {entry['fatal_recall_mean']:.3f} prec {entry['fatal_precision_mean']:.3f}")
    return {"rows_pre2026": int(len(df_m)), "fatal_share_pre2026": round(float(y_m.mean()), 5), "models": results}


def main(csv_path, out_json):
    t0 = time.time()
    df, X_base, y, year = rc.load_pre2026(csv_path)
    assert year.max() <= 2025, "2026 must never be loaded"
    rc.log(f"Loaded pre-2026 rows={len(df):,}, years={year.min()}-{year.max()}")

    main_results = []
    for label, harm, eng in (
        ("baseline_native_17", False, False),
        ("harmonized_2023_only", True, False),
        ("harmonized_plus_engineered", True, True),
    ):
        X_up, cats = build_upgraded_features(df, X_base, harmonize_2023=harm, add_engineered=eng)
        main_results.append(eval_multiclass_lgbm(X_up, y, year, cats, label))

    moto_results = eval_motorcycle_binary(df, X_base, y, year)

    payload = {
        "note": "Post hoc validation-only experiments on 2022-2025 rolling folds (2026 never loaded).",
        "elapsed_seconds": round(time.time() - t0, 1),
        "main_thai": main_results,
        "motorcycle_binary": moto_results,
    }
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    rc.log(f"Saved {out} in {payload['elapsed_seconds']}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
