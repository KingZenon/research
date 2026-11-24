"""Follow-up audit checks (no model): what does n_involved > n_vehicles mean, and how many 2023 labels are unseen on the modelled rows?

Usage:  python -X utf8 audit_extras.py <combined_csv> <results_recheck_dir>
Writes audit_extras.json.  2026 rows are dropped at load time.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import recheck_common as rc


def main(csv_path, out_dir):
    df, X, y, year = rc.load_pre2026(csv_path)
    res = {}
    extra = (X["n_involved"] - X["n_vehicles"]) > 0
    ped_hit = X["acc_type"].str.contains("ชนคนเดินเท้า", na=False)
    ped_veh = X["vehicle1"].eq("คนเดินเท้า")
    res["rows_with_extra_persons"] = int(extra.sum())
    res["share_of_extra_rows_with_acc_type_pedestrian_hit"] = round(float(ped_hit[extra].mean()), 4)
    res["share_of_all_rows_with_acc_type_pedestrian_hit"] = round(float(ped_hit.mean()), 4)
    res["share_of_pedestrian_hit_rows_with_extra_persons"] = round(float(extra[ped_hit].mean()), 4)
    res["fatal_share_pedestrian_hit_rows"] = round(float((y[ped_hit.to_numpy()] == 0).mean()), 4)
    res["fatal_share_extra_rows_not_pedestrian_hit"] = round(float((y[(extra & ~ped_hit).to_numpy()] == 0).mean()), 4)
    # relation of the extra persons to the casualties: if n_involved counted casualties it would grow with injured + deaths
    cas = (df["deaths"] + df["injured_total"]).to_numpy()
    nv = X["n_vehicles"].to_numpy()
    both = X["n_involved"].notna().to_numpy()
    res["mean_casualties_when_extra_persons_0_vs_positive"] = [round(float(cas[both & ~extra.to_numpy()].mean()), 3), round(float(cas[both & extra.to_numpy()].mean()), 3)]
    res["share_of_rows_with_casualties_greater_than_n_involved"] = round(float((cas[both] > X["n_involved"].to_numpy()[both]).mean()), 4)
    # 2023 label shares on the modelled rows (after the original duplicate rule), versus the original unseen-share file
    for col in ("cause", "location", "acc_type"):
        seen = set(X.loc[year <= 2022, col].unique())
        m = year == 2023
        res[f"unseen_share_2023_{col}_on_modelled_rows_incl_missing_label"] = round(float((~X.loc[m, col].isin(seen)).mean()), 4)
        nm = m & (X[col] != "missing").to_numpy()
        res[f"unseen_share_2023_{col}_on_modelled_rows_excl_missing"] = round(float((~X.loc[nm, col].isin(seen)).mean()), 4)
    # fatal share of rows with an unseen 2023 label vs seen label
    seen_loc = set(X.loc[year <= 2022, "location"].unique())
    m = (year == 2023)
    unseen = (~X.loc[m, "location"].isin(seen_loc)).to_numpy()
    res["fatal_share_2023_rows_with_unseen_location_label_vs_seen"] = [round(float((y[m][unseen] == 0).mean()), 4), round(float((y[m][~unseen] == 0).mean()), 4)]
    (Path(out_dir) / "audit_extras.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
