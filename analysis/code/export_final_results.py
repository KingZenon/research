"""Write the final-result numbers used by the project website into website/final-data.js.

Usage:  python -X utf8 export_final_results.py <combined_csv> <results_dir> <site_dir>

Reads the files written by aggregate_final.py and final_test_2026.py (selection report, the single 2026 test result
with bootstrap intervals and confusion matrix), the UK replication and the motorcycle subset, and writes
window.FINAL_DATA. No score is typed by hand. The only constants below are the benchmark figures quoted from
Mohamad et al. (2025), the paper the motorcycle comparison refers to.
"""
import json
import sys
from pathlib import Path

import pandas as pd

import baseline_models as bm

MODEL_LABELS = {"majority_class": "Majority class (baseline)", "logistic_regression": "Logistic regression",
                "decision_tree": "Decision tree", "random_forest": "Random forest", "xgboost": "XGBoost", "lightgbm": "LightGBM",
                "xgboost_tuned": "XGBoost (tuned)", "lightgbm_tuned": "LightGBM (tuned)", "random_forest_tuned": "Random forest (tuned)"}
VARIANT_LABELS = {"default": "Default", "class_weight": "Class weights", "prior_adjusted": "Prior-adjusted", "smote": "SMOTE-NC"}
BENCHMARK = {"label": "Benchmark: Mohamad et al. (2025), random forest", "source": "Mohamad et al. (2025), random forest, Thai motorcycle crashes 2015–2020",
             "accuracy": 0.777, "auc": 0.768, "fatalRecall": 0.198, "fatalShare": 0.244}
FAMILIES = ["logistic_regression", "decision_tree", "random_forest", "xgboost", "lightgbm"]


def r(x, d=5):
    return round(float(x), d)


def rows_of(summary, models, variants, keys):
    out = []
    for m in models:
        for v in variants:
            hit = summary[(summary.model == m) & (summary.variant == v)]
            if hit.empty:
                continue
            row = hit.iloc[0]
            entry = {"model": m, "label": MODEL_LABELS[m], "variant": v, "variantLabel": VARIANT_LABELS[v]}
            entry.update({k: r(row[f"{k}_mean"]) for k in keys})
            if "fatal_f1" in keys:
                entry["fatal_f1_std"] = r(row["fatal_f1_std"])
            out.append(entry)
    return out


def main(csv, results_dir, site_dir):
    R, site = Path(results_dir), Path(site_dir)
    thai = pd.read_csv(R / "cv_thai_all_summary.csv")
    thai_folds = pd.read_csv(R / "cv_thai_all_folds.csv")
    uk = pd.read_csv(R / "cv_uk_summary.csv")
    uk_folds = pd.read_csv(R / "cv_uk_folds.csv")
    moto = pd.read_csv(R / "cv_motorcycle_summary.csv")
    sel = json.loads((R / "final_selection.json").read_text(encoding="utf-8"))
    report = pd.read_csv(R / "selection_report.csv")
    test = pd.read_csv(R / "test_2026_final.csv").set_index("model")
    boot = pd.read_csv(R / "test_2026_bootstrap.csv").set_index("metric")
    conf = json.loads((R / "test_2026_confusion.json").read_text(encoding="utf-8"))
    audit = json.loads((R / "uk_audit.json").read_text(encoding="utf-8"))
    control = pd.read_csv(R / "cv_thai_smote_control_folds.csv").groupby("model")[["fatal_f1", "fatal_pr_auc", "fatal_recall"]].mean()
    tuning = pd.read_csv(R / "tuning_thai.csv")

    chosen = test.loc[f"{sel['model']}_tuned" if sel["tuned"] else sel["model"]]
    base = test.loc["majority_class"]
    metric_spec = [("fatal_recall", "Fatal recall"), ("fatal_precision", "Fatal precision"), ("fatal_f1", "Fatal F1"),
                   ("fatal_pr_auc", "Fatal PR-AUC"), ("fatal_auc", "Fatal ROC-AUC"), ("macro_f1", "Macro-F1"),
                   ("balanced_acc", "Balanced accuracy"), ("accuracy", "Accuracy")]
    metrics = []
    for key, label in metric_spec:
        entry = {"key": key, "label": label, "value": r(chosen[key]), "baseline": r(base[key])}
        if key in boot.index:
            entry.update({"ciLow": r(boot.loc[key, "ci_low"]), "ciHigh": r(boot.loc[key, "ci_high"])})
        metrics.append(entry)

    top = []
    for row in report.head(6).itertuples():
        top.append({"rank": int(row.rank), "model": row.model, "label": MODEL_LABELS[row.model], "variant": row.variant,
                    "variantLabel": VARIANT_LABELS[row.variant], "fatalF1": r(row.fatal_f1_mean), "fatalF1Std": r(row.fatal_f1_std),
                    "macroF1": r(row.macro_f1_mean), "recall": r(row.fatal_recall_mean), "precision": r(row.fatal_precision_mean),
                    "prAuc": r(row.fatal_pr_auc_mean), "gapToWinner": r(row.gap_to_winner_mean),
                    "winnerAhead": None if pd.isna(row.winner_ahead_in_folds) else int(row.winner_ahead_in_folds)})

    # effect of tuning on the outer folds
    def g(m, v, k):
        return float(thai[(thai.model == m) & (thai.variant == v)][f"{k}_mean"].iloc[0])
    inner = tuning.groupby(["model", "val_year"]).apply(lambda d: d["inner_pr_auc"].max() - d.loc[d.config_id == 0, "inner_pr_auc"].iloc[0], include_groups=False)
    tuning_rows = [{"family": MODEL_LABELS[m], "plainPrAuc": r(g(m, "default", "fatal_pr_auc")), "tunedPrAuc": r(g(f"{m}_tuned", "default", "fatal_pr_auc")),
                    "weightsF1": r(g(m, "class_weight", "fatal_f1")), "tunedWeightsF1": r(g(f"{m}_tuned", "class_weight", "fatal_f1"))}
                   for m in ("xgboost", "lightgbm")]
    smote_control = {m: {"controlF1": r(control.loc[f"{m}_imputed_control", "fatal_f1"]), "controlPrAuc": r(control.loc[f"{m}_imputed_control", "fatal_pr_auc"]),
                         "plainF1": r(g(m, "default", "fatal_f1")), "plainPrAuc": r(g(m, "default", "fatal_pr_auc")),
                         "smoteF1": r(g(m, "smote", "fatal_f1")), "smotePrAuc": r(g(m, "smote", "fatal_pr_auc")),
                         "plainRecall": r(g(m, "default", "fatal_recall")), "smoteRecall": r(g(m, "smote", "fatal_recall"))}
                     for m in ("xgboost", "lightgbm")}

    uk_keys = ["fatal_recall", "fatal_precision", "fatal_f1", "fatal_pr_auc", "fatal_auc", "macro_f1"]
    th_keys = uk_keys
    mean_prev_th = float(thai_folds[(thai_folds.model == "lightgbm_tuned") & (thai_folds.variant == "default")]["fatal_share_val"].mean())
    mean_prev_uk = float(uk_folds[uk_folds.model == "majority_class"]["fatal_share_val"].mean())

    mdf = bm.load(csv, "motorcycle")
    mdf = mdf[mdf["acc_year"].notna()]
    mkeys = ["fatal_vs_rest_accuracy", "fatal_auc", "fatal_recall", "fatal_precision", "fatal_f1"]

    payload = {
        "source": "Generated by analysis/code/export_final_results.py from the files in analysis/results (aggregate_final.py and final_test_2026.py outputs).",
        "selection": {"rule": sel["rule"], "model": MODEL_LABELS[sel["model"] + ("_tuned" if sel["tuned"] else "")], "variantLabel": VARIANT_LABELS[sel["variant"]],
                      "fatalF1": sel["fatal_f1_mean"], "candidates": sel["candidates_compared"], "top": top},
        "test": {"year": int(chosen["test_year"]), "trainYears": str(chosen["train_years"]).replace("-", "–"), "nTrain": int(chosen["n_train"]),
                 "nTest": int(chosen["n_test"]), "fatalShare": r(chosen["fatal_share_test"]), "params": json.loads(chosen["params"]),
                 "metrics": metrics, "classes": ["Fatal", "Injury", "Property damage only"], "confusion": conf["matrix"]},
        "tuning": {"innerGainMin": r(inner.min()), "innerGainMax": r(inner.max()), "families": tuning_rows, "randomConfigs": int(tuning.groupby(["model", "val_year"]).size().iloc[0] - 1)},
        "smoteControl": smote_control,
        "thai": {"fatalShareValidation": r(mean_prev_th),
                 "lightgbm": rows_of(thai, ["lightgbm"], list(VARIANT_LABELS), th_keys)},
        "uk": {"rows": audit["rows"], "years": audit["years"], "classShare": audit["classShare"], "fatalShareByYear": audit["fatalShareByYear"],
               "folds": int(uk["n_folds"].iloc[0]), "fatalShareValidation": r(mean_prev_uk),
               "summary": rows_of(uk, ["majority_class"] + FAMILIES, list(VARIANT_LABELS), uk_keys)},
        "motorcycle": {"rows": int(len(mdf)), "fatalShare": r((mdf["severity"] == "fatal").mean()), "benchmark": BENCHMARK,
                       "summary": rows_of(moto, ["majority_class"] + FAMILIES + ["random_forest_tuned", "xgboost_tuned", "lightgbm_tuned"],
                                           list(VARIANT_LABELS), mkeys)},
    }
    # post hoc re-check (results_recheck): native categories, scaled SMOTE, calibration, inner-year threshold, feature timing, subgroups
    RCD = R.parent / "results_recheck"
    if RCD.exists():
        log_rc = pd.read_csv(RCD / "experiment_log.csv").drop_duplicates(subset=["stage", "family", "encoding", "strategy", "val_year"], keep="last")

        def rcv(stage, family, encoding, strategy, metric):
            d = log_rc[(log_rc.stage == stage) & (log_rc.family == family) & (log_rc.encoding == encoding) & (log_rc.strategy == strategy)].sort_values("val_year")
            assert len(d) == 4, (stage, family, encoding, strategy, len(d))
            return d[metric].to_numpy()

        cal = pd.read_csv(RCD / "calibration_summary.csv").set_index(["family", "encoding", "strategy"])
        sub_rc = pd.read_csv(RCD / "err2025_lightgbm_native_thr_f1_inner_subgroup_recall.csv", encoding="utf-8-sig")
        prof = pd.read_csv(RCD / "err2025_lightgbm_native_thr_f1_inner_error_profile.csv", encoding="utf-8-sig")
        thr = rcv("exp4_threshold", "lightgbm", "native", "thr_f1_inner", "fatal_f1")
        cw = rcv("exp1_native_vs_ordinal", "lightgbm", "native", "class_weight", "fatal_f1")

        def subg(dim, group):
            row = sub_rc[(sub_rc.dimension == dim) & (sub_rc.group == group)].iloc[0]
            return {"nFatal": int(row["n_fatal"]), "recall": r(row["fatal_recall"])}

        def share(kind, col, label):
            d = prof[prof.kind.str.startswith(kind) & (prof.column == col) & (prof.label == label)]
            return r(d.share_in_errors.iloc[0])

        e1 = "exp1_native_vs_ordinal"
        payload["recheck"] = {
            "note": "A post hoc re-check on the same validation years with stricter practice. The 2026 test year was not used.",
            "native": [{"family": label, "plainPrAucOrdinal": r(rcv(e1, fam, "ordinal", "plain", "fatal_pr_auc").mean()),
                        "plainPrAucNative": r(rcv(e1, fam, "native", "plain", "fatal_pr_auc").mean())} for label, fam in (("LightGBM", "lightgbm"), ("XGBoost", "xgboost"))],
            "smote": [{"family": label, "control": r(rcv("exp2_smote", fam, "control_ordinal", "plain", "fatal_pr_auc").mean()),
                       "unscaled": r(g(fam, "smote", "fatal_pr_auc")),
                       "scaled": r(rcv("exp2_smote", fam, "smote_scaled_ordinal", "plain", "fatal_pr_auc").mean())} for label, fam in (("LightGBM", "lightgbm"), ("XGBoost", "xgboost"))],
            "calibration": [{"label": label, "ece": r(cal.loc[("lightgbm", "native", strat), "ece_10_equal_width_mean"])}
                            for label, strat in (("Plain", "plain"), ("Class weights", "class_weight"), ("Prior-adjusted", "prior_adjusted"))],
            "threshold": {"f1": r(thr.mean()), "sd": r(thr.std(ddof=1)), "weightsF1": r(cw.mean()), "foldsAhead": int((thr > cw).sum()),
                          "recall": r(rcv("exp4_threshold", "lightgbm", "native", "thr_f1_inner", "fatal_recall").mean()),
                          "precision": r(rcv("exp4_threshold", "lightgbm", "native", "thr_f1_inner", "fatal_precision").mean())},
            "timing": {"all": r(rcv(e1, "lightgbm", "native", "plain", "fatal_pr_auc").mean()),
                       "without": r(rcv("exp6_feature_ablation", "lightgbm", "native__no_cause_acctype_ninvolved", "plain", "fatal_pr_auc").mean())},
            "subgroups": [dict(label=label, **subg(dim, grp)) for label, dim, grp in (
                ("Motorcycle", "vehicle1", "motorcycle"), ("Pickup / 4-wheel truck", "vehicle1", "pickup/4-wheel truck"), ("Car", "vehicle1", "car (private/public)"),
                ("Northeast region", "region", "Northeast"), ("North region", "region", "North"),
                ("Night (18:00\u201306:00)", "night(18-06)", "night"), ("Day", "night(18-06)", "day"))],
            "errors": {"fnPickup": share("false_negative", "vehicle", "pickup/4-wheel truck"), "fnCar": share("false_negative", "vehicle", "car (private/public)"),
                       "fnMotorcycle": share("false_negative", "vehicle", "motorcycle"), "fpMotorcycle": share("false_positive", "vehicle", "motorcycle")},
        }
        exp11_path = RCD / "recheck_exp11_improvements.json"
        if not exp11_path.exists():
            exp11_path = R / "recheck_exp11_improvements.json"
        if exp11_path.exists():
            payload["recheck"]["exp11"] = json.loads(exp11_path.read_text(encoding="utf-8"))
    text = ("/* Generated file: do not edit by hand. Run analysis/code/export_final_results.py to rebuild. */\n"
            "window.FINAL_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n")
    (site / "final-data.js").write_text(text, encoding="utf-8")
    print("wrote", site / "final-data.js", f"{len(text) / 1024:.0f} KB")


if __name__ == "__main__":
    main(*sys.argv[1:4])
