"""Experiment 5: error analysis on the 2025 validation fold (train 2019-2024) for ONE chosen model.

Usage:  python -X utf8 exp5_error_analysis.py <combined_csv> <results_recheck_dir> --family lightgbm --encoding native
                                              --strategy class_weight|plain|prior_adjusted|thr_f1_inner [--repeats 5]

Outputs (results_recheck_dir, prefix err2025_<family>_<encoding>_<strategy>):
  _confusion.csv                 counts and row percentages (rows actual, columns predicted: fatal / injury / property_only)
  _pr_curve.csv, _pr_curve.png   fatal precision-recall curve of the model (+ plain vs class-weighted overlay of the same family)
  _subgroup_recall.csv, .png     fatal recall / precision / n by vehicle type, region, 3-hour band, day/night, weather, agency,
                                 and by validation year 2022-2025 (from the saved fold probabilities)
  _worst_errors.csv, _error_profile.csv   30 worst false negatives and false positives; what the 300 worst of each have in common
  _perm_importance.csv, .png     grouped permutation importance (groups of near-duplicate columns are permuted jointly)
The casualty columns appear ONLY in this descriptive analysis (never as features). 2026 is never loaded.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

import recheck_common as rc

REGIONS = {
    "North": "เชียงราย เชียงใหม่ น่าน พะเยา แพร่ แม่ฮ่องสอน ลำปาง ลำพูน อุตรดิตถ์",
    "Northeast": "กาฬสินธุ์ ขอนแก่น ชัยภูมิ นครพนม นครราชสีมา บึงกาฬ บุรีรัมย์ มหาสารคาม มุกดาหาร ยโสธร ร้อยเอ็ด เลย สกลนคร สุรินทร์ ศรีสะเกษ หนองคาย หนองบัวลำภู อุดรธานี อุบลราชธานี อำนาจเจริญ",
    "Central": "กรุงเทพมหานคร กำแพงเพชร ชัยนาท นครนายก นครปฐม นครสวรรค์ นนทบุรี ปทุมธานี อยุธยา พิจิตร พิษณุโลก เพชรบูรณ์ ลพบุรี สมุทรปราการ สมุทรสงคราม สมุทรสาคร สิงห์บุรี สุโขทัย สุพรรณบุรี สระบุรี อ่างทอง อุทัยธานี",
    "West": "กาญจนบุรี ตาก ประจวบคีรีขันธ์ เพชรบุรี ราชบุรี",
    "East": "จันทบุรี ฉะเชิงเทรา ชลบุรี ตราด ปราจีนบุรี ระยอง สระแก้ว",
    "South": "กระบี่ ชุมพร ตรัง นครศรีธรรมราช นราธิวาส ปัตตานี พังงา พัทลุง ภูเก็ต ยะลา ระนอง สงขลา สตูล สุราษฎร์ธานี",
}
PROVINCE_TO_REGION = {p: r for r, names in REGIONS.items() for p in names.split()}
GROUPS = {"n_vehicles+n_involved": ["n_vehicles", "n_involved"], "lat+lon": ["lat", "lon"], "km": ["km"],
          "hour+is_night": ["hour", "is_night"], "month": ["month"], "weekday+is_weekend": ["weekday", "is_weekend"],
          "province": ["province"], "agency": ["agency"], "vehicle1": ["vehicle1"], "location": ["location"],
          "cause": ["cause"], "acc_type": ["acc_type"], "weather": ["weather"],
          "ALL_PLACE (province+lat+lon+km+agency)": ["province", "lat", "lon", "km", "agency"]}
VEHICLE_EN = {"รถปิคอัพบรรทุก4ล้อ": "pickup/4-wheel truck", "รถยนต์นั่งส่วนบุคคล/รถยนต์นั่งสาธารณะ": "car (private/public)",
              "รถจักรยานยนต์": "motorcycle", "รถบรรทุกมากกว่า10ล้อ(รถพ่วง)": "truck >10 wheels (trailer)", "รถบรรทุก6ล้อ": "6-wheel truck",
              "อื่นๆ": "other", "รถบรรทุกมากกว่า6ล้อไม่เกิน10ล้อ": "truck 6-10 wheels", "รถตู้": "van", "รถโดยสารขนาดใหญ่": "large bus",
              "รถปิคอัพโดยสาร": "passenger pickup", "รถสามล้อเครื่อง": "motor tricycle", "รถจักรยาน": "bicycle",
              "คนเดินเท้า": "pedestrian", "รถอีแต๋น/เพื่อการเกษตร": "agricultural vehicle", "รถสามล้อ": "tricycle", "missing": "missing"}
AGENCY_EN = {"กรมทางหลวง": "DOH (highways)", "กรมทางหลวงชนบท": "DRR (rural roads)", "การทางพิเศษแห่งประเทศไทย": "EXAT (expressways)"}


class Model:
    """Fitted model + the encoding it needs, so permutation code can call predict_proba on raw feature frames."""

    def __init__(self, family, native, variant, X_tr, y_tr):
        self.native, self.family = native, family
        self.cats = rc.fit_categories(X_tr)
        if native:
            from sklearn.utils.class_weight import compute_sample_weight
            self.clf = rc.make_clf(family, True, variant)
            kw = {"sample_weight": compute_sample_weight("balanced", y_tr)} if (variant == "class_weight" and family == "xgboost") else {}
            self.clf.fit(rc.to_native(X_tr, self.cats), y_tr, **kw)
        else:
            self.clf = rc.cv.fit_variant(family, variant, X_tr, y_tr, rc.CAT, rc.NUM, {})

    def predict_proba(self, X):
        return self.clf.predict_proba(rc.to_native(X, self.cats) if self.native else X)


def decide(proba, strategy, prior, thr):
    if strategy == "prior_adjusted":
        return (proba / prior).argmax(1)
    if strategy.startswith("thr_"):
        return rc.predict_with_threshold(proba, thr)
    return proba.argmax(1)


def recall_table(label, groups, pred, y, rows):
    for g in pd.unique(groups):
        m = np.asarray(groups == g)
        n_fatal = int(((y == 0) & m).sum())
        if m.sum() < 30:
            continue
        rows.append({"dimension": label, "group": g, "n": int(m.sum()), "n_fatal": n_fatal, "fatal_share": n_fatal / m.sum(),
                     "fatal_recall": float(((pred == 0) & (y == 0) & m).sum() / n_fatal) if n_fatal else np.nan,
                     "fatal_precision": float(((pred == 0) & (y == 0) & m).sum() / max(((pred == 0) & m).sum(), 1)),
                     "predicted_fatal_share": float(((pred == 0) & m).sum() / m.sum())})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("results_dir")
    ap.add_argument("--family", required=True)
    ap.add_argument("--encoding", required=True, choices=["ordinal", "native"])
    ap.add_argument("--strategy", required=True, choices=["plain", "class_weight", "prior_adjusted", "thr_f1_inner"])
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    out = Path(args.results_dir)
    tag = f"err2025_{args.family}_{args.encoding}_{args.strategy}"
    native = args.encoding == "native"
    variant = "class_weight" if args.strategy == "class_weight" else "default"

    df, X, y, year = rc.load_pre2026(args.csv)
    tr, va = year <= 2024, year == 2025
    Xv, yv = X[va].reset_index(drop=True), y[va]
    dv = df[va].reset_index(drop=True)
    prior = np.bincount(y[tr], minlength=3) / tr.sum()
    log = pd.read_csv(out / "experiment_log.csv")

    def threshold_for(val_year):
        if not args.strategy.startswith("thr_"):
            return None
        r = log[(log.stage == "exp4_threshold") & (log.family == args.family) & (log.encoding == args.encoding)
                & (log.strategy == args.strategy) & (log.val_year == val_year)]
        return float(r["threshold"].iloc[0])

    rc.log(f"fitting {args.family} {args.encoding} {variant} on 2019-2024 ({int(tr.sum()):,} rows)")
    model = Model(args.family, native, variant, X[tr], y[tr])
    proba = model.predict_proba(Xv)
    pred = decide(proba, args.strategy, prior, threshold_for(2025))
    score = proba[:, 0]
    # consistency with the experiment log (same fit, same seed)
    logged = log[(log.family == args.family) & (log.encoding == args.encoding) & (log.strategy == args.strategy) & (log.val_year == 2025)
                 & (log.stage.isin(["exp1_native_vs_ordinal", "exp4_threshold"]))]
    f1_now = f1_score(yv, pred, labels=[0], average=None)[0]
    rc.log(f"fatal F1 on 2025 now {f1_now:.4f}; logged {logged['fatal_f1'].iloc[0]:.4f}")

    # 1. confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(yv, pred, labels=[0, 1, 2])
    cmd = pd.DataFrame(cm, index=[f"actual_{c}" for c in rc.bm.CLASSES], columns=[f"pred_{c}" for c in rc.bm.CLASSES])
    pct = (cm / cm.sum(axis=1, keepdims=True) * 100).round(1)
    cmp_ = pd.DataFrame(pct, index=cmd.index, columns=[c + "_row_pct" for c in cmd.columns])
    pd.concat([cmd, cmp_], axis=1).to_csv(out / f"{tag}_confusion.csv")
    print(pd.concat([cmd, cmp_], axis=1).to_string())

    # 2. PR curve (+ plain vs class_weight overlay of the same family / encoding from the saved probabilities)
    prec, rec, thr = precision_recall_curve((yv == 0).astype(int), score)
    pd.DataFrame({"precision": prec[:-1], "recall": rec[:-1], "threshold": thr}).iloc[::20].to_csv(out / f"{tag}_pr_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4.8))
    for nm in ("plain", "class_weight"):
        _, pp = rc.load_proba(out, f"exp1_native_vs_ordinal_{args.family}_{args.encoding}_{nm}", 2025)
        a, b, _ = precision_recall_curve((yv == 0).astype(int), pp[:, 0])
        ax.plot(b, a, label=f"{nm} (AP {average_precision_score((yv == 0).astype(int), pp[:, 0]):.3f})", lw=1.5, alpha=.85)
    op_rec = cm[0, 0] / cm[0].sum()
    op_prec = cm[0, 0] / max(cm[:, 0].sum(), 1)
    ax.plot([op_rec], [op_prec], "k*", ms=11, label=f"chosen rule: {args.strategy} (R {op_rec:.2f}, P {op_prec:.2f})")
    ax.axhline((yv == 0).mean(), color="grey", ls=":", label=f"prevalence {np.mean(yv == 0):.3f}")
    ax.set_xlabel("fatal recall")
    ax.set_ylabel("fatal precision")
    ax.set_title(f"2025 validation, {args.family} {args.encoding}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{tag}_pr_curve.png", dpi=140)
    plt.close(fig)

    # 3. subgroups
    rows = []
    recall_table("vehicle1", Xv["vehicle1"].map(lambda v: VEHICLE_EN.get(v, v)).to_numpy(), pred, yv, rows)
    recall_table("region", Xv["province"].map(lambda p: PROVINCE_TO_REGION.get(p, "unmapped/missing")).to_numpy(), pred, yv, rows)
    band = pd.cut(Xv["hour"], [-1, 2, 5, 8, 11, 14, 17, 20, 23], labels=["00-02", "03-05", "06-08", "09-11", "12-14", "15-17", "18-20", "21-23"]).astype(str)
    recall_table("hour_band", band.to_numpy(), pred, yv, rows)
    recall_table("night(18-06)", np.where(Xv["is_night"] == 1, "night", "day").astype(object), pred, yv, rows)
    recall_table("weather", Xv["weather"].to_numpy(), pred, yv, rows)
    recall_table("agency", Xv["agency"].map(lambda v: AGENCY_EN.get(v, v)).to_numpy(), pred, yv, rows)
    recall_table("n_vehicles", np.where(Xv["n_vehicles"] >= 3, "3+", np.where(Xv["n_vehicles"] == 2, "2", "1")).astype(object), pred, yv, rows)
    # validation year: every fold with the same decision rule
    for last_train, val_year in rc.FOLDS:
        trf, vaf = year <= last_train, year == val_year
        _, pf = rc.load_proba(out, f"exp1_native_vs_ordinal_{args.family}_{args.encoding}_{'class_weight' if args.strategy == 'class_weight' else 'plain'}", val_year)
        pr = decide(pf, args.strategy, np.bincount(y[trf], minlength=3) / trf.sum(), threshold_for(val_year))
        recall_table("validation_year", np.array([str(val_year)] * int(vaf.sum()), dtype=object), pr, y[vaf], rows)
    sub = pd.DataFrame(rows)
    sub.to_csv(out / f"{tag}_subgroup_recall.csv", index=False, encoding="utf-8-sig")
    dims = ["vehicle1", "region", "hour_band", "validation_year"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, dim in zip(axes.ravel(), dims):
        d = sub[sub.dimension == dim]
        if dim in ("vehicle1", "region"):
            d = d.sort_values("fatal_recall")
        ax.barh(d["group"].astype(str), d["fatal_recall"], color="#3b7dd8")
        ax.axvline(op_rec, color="k", ls=":")
        ax.set_title(f"fatal recall by {dim} (dotted: overall {op_rec:.2f})")
        ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(out / f"{tag}_subgroup_recall.png", dpi=140)
    plt.close(fig)

    # 4. worst errors
    base = pd.DataFrame({"p_fatal": score, "actual": [rc.bm.CLASSES[i] for i in yv], "predicted": [rc.bm.CLASSES[i] for i in pred],
                         "vehicle": Xv["vehicle1"].map(lambda v: VEHICLE_EN.get(v, v)), "cause": Xv["cause"], "acc_type": Xv["acc_type"],
                         "location": Xv["location"], "region": Xv["province"].map(lambda p: PROVINCE_TO_REGION.get(p, "unmapped/missing")),
                         "hour": Xv["hour"], "n_vehicles": Xv["n_vehicles"], "n_involved": Xv["n_involved"], "weather": Xv["weather"],
                         "agency": Xv["agency"].map(lambda v: AGENCY_EN.get(v, v)),
                         "deaths": dv["deaths"], "serious": dv["serious"], "minor": dv["minor"]})
    fn = base[(yv == 0) & (pred != 0)].sort_values("p_fatal")
    fp = base[(yv != 0) & (pred == 0)].sort_values("p_fatal", ascending=False)
    worst = pd.concat([fn.head(30).assign(kind="false_negative"), fp.head(30).assign(kind="false_positive")])
    worst.to_csv(out / f"{tag}_worst_errors.csv", index=False, encoding="utf-8-sig")
    profile = []
    for kind, part, ref in (("false_negative(300 lowest P)", fn.head(300), base[yv == 0]), ("false_positive(300 highest P)", fp.head(300), base[yv != 0])):
        for col in ("vehicle", "cause", "acc_type", "location", "region", "weather", "agency"):
            share = part[col].value_counts(normalize=True).head(4)
            for lab, s in share.items():
                refshare = float((ref[col] == lab).mean())
                profile.append({"kind": kind, "column": col, "label": lab, "share_in_errors": round(float(s), 3), "share_in_reference": round(refshare, 3),
                                "lift": round(float(s) / refshare, 2) if refshare > 0 else np.nan})
        for col in ("hour", "n_vehicles", "n_involved"):
            profile.append({"kind": kind, "column": col + "_mean", "label": "", "share_in_errors": round(float(part[col].mean()), 3),
                            "share_in_reference": round(float(ref[col].mean()), 3), "lift": np.nan})
    profile.append({"kind": "false_negative(300 lowest P)", "column": "deaths_mean", "label": "", "share_in_errors": round(float(fn.head(300)["deaths"].mean()), 3),
                    "share_in_reference": round(float(base[yv == 0]["deaths"].mean()), 3), "lift": np.nan})
    profile.append({"kind": "false_positive(300 highest P)", "column": "serious_mean", "label": "", "share_in_errors": round(float(fp.head(300)["serious"].mean()), 3),
                    "share_in_reference": round(float(base[yv != 0]["serious"].mean()), 3), "lift": np.nan})
    profile.append({"kind": "false_positive(300 highest P)", "column": "share_actually_injury_not_property", "label": "",
                    "share_in_errors": round(float((fp.head(300)["actual"] == "injury").mean()), 3), "share_in_reference": round(float((base[yv != 0]["actual"] == "injury").mean()), 3), "lift": np.nan})
    pd.DataFrame(profile).to_csv(out / f"{tag}_error_profile.csv", index=False, encoding="utf-8-sig")
    rc.log(f"false negatives {len(fn)}, false positives {len(fp)}; FN P(fatal) range {fn.p_fatal.min():.4f}-{fn.p_fatal.max():.4f}")

    # 5. grouped permutation importance
    rng = np.random.default_rng(rc.SEED)
    base_ap = average_precision_score((yv == 0).astype(int), score)
    base_f1 = f1_now
    res = []
    for gname, cols in GROUPS.items():
        d_ap, d_f1 = [], []
        for _ in range(args.repeats):
            idx = rng.permutation(len(Xv))
            Xp = Xv.copy()
            Xp[cols] = Xv[cols].iloc[idx].reset_index(drop=True)
            pp = model.predict_proba(Xp)
            d_ap.append(base_ap - average_precision_score((yv == 0).astype(int), pp[:, 0]))
            d_f1.append(base_f1 - f1_score(yv, decide(pp, args.strategy, prior, threshold_for(2025)), labels=[0], average=None)[0])
        res.append({"group": gname, "columns": "+".join(cols), "pr_auc_drop_mean": np.mean(d_ap), "pr_auc_drop_sd": np.std(d_ap, ddof=1),
                    "fatal_f1_drop_mean": np.mean(d_f1), "fatal_f1_drop_sd": np.std(d_f1, ddof=1)})
        rc.log(f"  permuted {gname:42s} PR-AUC drop {np.mean(d_ap):+.4f}  F1 drop {np.mean(d_f1):+.4f}")
    imp = pd.DataFrame(res).sort_values("pr_auc_drop_mean", ascending=False)
    imp.insert(0, "baseline_pr_auc", base_ap)
    imp.to_csv(out / f"{tag}_perm_importance.csv", index=False)
    corr = {"spearman_n_vehicles_n_involved": float(Xv["n_vehicles"].corr(Xv["n_involved"], method="spearman")),
            "spearman_lat_lon": float(Xv["lat"].corr(Xv["lon"], method="spearman")),
            "spearman_hour_is_night": float(Xv["hour"].corr(Xv["is_night"], method="spearman"))}
    pd.Series(corr).to_csv(out / f"{tag}_group_correlations.csv", header=["value"])
    fig, ax = plt.subplots(figsize=(8, 5))
    d = imp.sort_values("pr_auc_drop_mean")
    ax.barh(d["group"], d["pr_auc_drop_mean"], xerr=d["pr_auc_drop_sd"], color="#3b7dd8")
    ax.set_xlabel("drop in fatal PR-AUC when the group is permuted (2025 validation)")
    fig.tight_layout()
    fig.savefig(out / f"{tag}_perm_importance.png", dpi=140)
    plt.close(fig)
    pd.set_option("display.width", 250)
    print(imp.round(4).to_string(index=False))
    print(sub[sub.dimension.isin(["vehicle1", "region", "hour_band", "validation_year", "night(18-06)"])].round(3).to_string(index=False))
    print(pd.DataFrame(profile).to_string(index=False))


if __name__ == "__main__":
    main()
