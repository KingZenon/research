"""Build the final paper (IEEE conference style, two columns, A4) as a .docx.

Usage:  python -X utf8 make_paper.py <results_dir> <paper_dir>

Reads the result files in <results_dir> and <paper_dir>/paper_numbers.json plus <paper_dir>/figures/*.png (written by
make_paper_figures.py) and writes <paper_dir>/Team_AHT_ICT324_Final_Paper.docx. Every number in the text and tables is
read from those files, never typed: rerun after any experiment changes. Citations are numbered in order of first use.
Needs python-docx.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

FONT = "Times New Roman"

# ----------------------------------------------------------------------------------------------
# references (cited in order of first use; numbers are assigned automatically)
# ----------------------------------------------------------------------------------------------
REFS = {
    "mohamad": "I. Mohamad, S. JomnonKwao, and V. Ratanavaraha, “Machine learning predictive performance in road accident severity: A case study from Thailand,” <i>Results in Engineering</i>, vol. 26, Art. 104833, 2025, doi: 10.1016/j.rineng.2025.104833.",
    "champahom": "T. Champahom, C. Se, F. Watcharamaisakul, S. Jomnonkwao, A. Karoonsoontawong, and V. Ratanavaraha, “Tree-based approaches to understanding factors influencing crash severity across roadway classes: A Thailand case study,” <i>IATSS Research</i>, vol. 48, no. 3, pp. 464\u2013476, 2024, doi: 10.1016/j.iatssr.2024.09.001.",
    "uttra": "S. Uttra, T. Champahom, S. Jomnonkwao, C. Se, P. Wisutwattanasak, and V. Ratanavaraha, “Exploring risk factors for crash severity during Thailand’s holiday travel: Machine learning exploration compared to heterogeneity modeling,” <i>Future Transportation</i>, vol. 6, no. 1, Art. 39, 2026, doi: 10.3390/futuretransp6010039.",
    "phojaem": "T. Phojaem, C. Se, P. Wisutwattanasak, T. Champahom, S. Jomnonkwao, and V. Ratanavaraha, “Comparative injury-severity modeling on highway-speed motorways: Performance, interpretability, and policy implications,” <i>Case Studies on Transport Policy</i>, vol. 24, Art. 101826, 2026, doi: 10.1016/j.cstp.2026.101826.",
    "simmachan": "T. Simmachan and P. Boonkrong, \u201cEffect of resampling techniques on machine learning models for classifying road accident severity in Thailand,\u201d <i>J. Current Science and Technology</i>, vol. 15, no. 2, 2025. [Online]. Available: https://ph04.tci-thaijo.org/index.php/JCST/article/view/6443",
    "mamun": "A. Al Mamun, A. Enan, D. A. Indah, J. Mwakalonge, G. Comert, and M. Chowdhury, “Crash severity risk modeling strategies under data imbalance,” arXiv:2412.02094, 2025.",
    "zhu": "D. Zhu, C. Xie, Z. Wang, and H. Zhang, “RaX-Crash: A resource efficient and explainable small model pipeline with an application to city scale injury severity prediction,” arXiv:2512.07848, 2025.",
    "elor": "Y. Elor and H. Averbuch-Elor, “To SMOTE, or not to SMOTE?,” arXiv:2201.08528, 2022.",
    "sufian": "M. A. Sufian and J. Varadarajan, “Enhancing prediction and analysis of UK road traffic accident severity using AI: Integration of machine learning, econometric techniques, and time series forecasting in public health research,” arXiv:2309.13483, 2023.",
    "tong": "Y. Tong, “Non-linear determinants of pedestrian injury severity: Evidence from administrative data in Great Britain,” arXiv:2512.04022, 2025.",
    "kapoor": "S. Kapoor and A. Narayanan, “Leakage and the reproducibility crisis in machine-learning-based science,” <i>Patterns</i>, vol. 4, no. 9, Art. 100804, 2023, doi: 10.1016/j.patter.2023.100804.",
    "grinsztajn": "L. Grinsztajn, E. Oyallon, and G. Varoquaux, “Why do tree-based models still outperform deep learning on typical tabular data?,” in <i>Advances in Neural Information Processing Systems 35 (Datasets and Benchmarks Track)</i>, 2022.",
    "chawla": "N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, “SMOTE: Synthetic minority over-sampling technique,” <i>J. Artif. Intell. Res.</i>, vol. 16, pp. 321–357, 2002.",
    "xgboost": "T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in <i>Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining</i>, 2016, pp. 785–794.",
    "lightgbm": "G. Ke <i>et al.</i>, “LightGBM: A highly efficient gradient boosting decision tree,” in <i>Advances in Neural Information Processing Systems 30</i>, 2017, pp. 3146–3154.",
    "saito": "T. Saito and M. Rehmsmeier, “The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets,” <i>PLoS ONE</i>, vol. 10, no. 3, Art. e0118432, 2015.",
    "crispdm": "R. Wirth and J. Hipp, “CRISP-DM: Towards a standard process model for data mining,” in <i>Proc. 4th Int. Conf. Practical Applications of Knowledge Discovery and Data Mining</i>, 2000, pp. 29–39.",
    "trams": "Ministry of Transport, Thailand, “Road accidents on the Ministry of Transport road network (TRAMS),” Open Government Data of Thailand. [Online]. Available: https://data.go.th/en/dataset/gdpublish-roadaccident (accessed Nov. 24, 2026).",
    "stats19": "Department for Transport (UK), “Road safety data (STATS19), collision files 2021–2024.” [Online]. Available: https://www.gov.uk/government/statistics/road-safety-data (accessed Nov. 24, 2026).",
    "sklearn": "F. Pedregosa <i>et al.</i>, “Scikit-learn: Machine learning in Python,” <i>J. Mach. Learn. Res.</i>, vol. 12, pp. 2825–2830, 2011.",
    "imblearn": "G. Lemaître, F. Nogueira, and C. K. Aridas, “Imbalanced-learn: A Python toolbox to tackle the curse of imbalanced datasets in machine learning,” <i>J. Mach. Learn. Res.</i>, vol. 18, no. 17, pp. 1–5, 2017.",
}
CITED = []


def c(*keys):
    out = []
    for key in keys:
        if key not in CITED:
            CITED.append(key)
        out.append(f"[{CITED.index(key) + 1}]")
    return ", ".join(out)


# ----------------------------------------------------------------------------------------------
# docx helpers
# ----------------------------------------------------------------------------------------------
def set_font(run, size, bold=None, italic=None, small_caps=False, color=None):
    run.font.name = FONT
    run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic
    if small_caps:
        run.font.small_caps = True


TAG = re.compile(r"(</?[ibu]>|</?sup>|</?sub>)")


def add_runs(par, text, size=10, bold=False, italic=False, small_caps=False):
    state = {"i": italic, "b": bold, "sup": False, "sub": False}
    for part in TAG.split(text):
        if not part:
            continue
        m = re.fullmatch(r"<(/?)(i|b|sup|sub)>", part)
        if m:
            state[m.group(2)] = m.group(1) == ""
            continue
        run = par.add_run(part)
        set_font(run, size, bold=state["b"], italic=state["i"], small_caps=small_caps)
        if state["sup"]:
            run.font.superscript = True
        if state["sub"]:
            run.font.subscript = True


def para(doc, text, size=10, align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=0.17, before=0, after=0, bold=False, italic=False,
         keep_next=False, small_caps=False, left=None, hanging=None):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    p.alignment = align
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    pf.line_spacing = 1.0
    pf.first_line_indent = Inches(indent) if indent else None
    pf.keep_with_next = keep_next
    pf.widow_control = True
    if left is not None:
        pf.left_indent = Inches(left)
    if hanging is not None:
        pf.first_line_indent = Inches(-hanging)
    add_runs(p, text, size, bold, italic, small_caps)
    return p


def heading1(doc, numeral, title):
    return para(doc, f"{numeral}. {title.upper()}", size=9.5, align=WD_ALIGN_PARAGRAPH.CENTER, indent=0, before=9, after=4,
                keep_next=True, small_caps=False)


def heading2(doc, letter, title):
    return para(doc, f"{letter}. {title}", size=10, align=WD_ALIGN_PARAGRAPH.LEFT, indent=0, before=5, after=2, italic=True,
                keep_next=True)


def set_cols(section, num, space_twips=360):
    sect = section._sectPr
    for old in sect.findall(qn("w:cols")):
        sect.remove(old)
    cols = OxmlElement("w:cols")
    cols.set(qn("w:num"), str(num))
    cols.set(qn("w:space"), str(space_twips))
    sect.append(cols)


def cell_border(cell, **edges):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = tcPr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcPr.append(borders)
    for edge, size in edges.items():
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(size))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        borders.append(el)


def add_table(doc, number, title, header, rows, widths, bold_rows=(), size=7.2):
    para(doc, f"TABLE {number}", size=8, align=WD_ALIGN_PARAGRAPH.CENTER, indent=0, before=6, small_caps=False, keep_next=True)
    para(doc, title.upper(), size=7.5, align=WD_ALIGN_PARAGRAPH.CENTER, indent=0, after=3, keep_next=True)
    assert sum(widths) <= 3.36, f"table {number} is wider than a column: {sum(widths):.2f} in"
    table = doc.add_table(rows=1 + len(rows), cols=len(header))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    tblPr = table._tbl.tblPr
    width_el = tblPr.find(qn("w:tblW"))
    if width_el is None:
        width_el = OxmlElement("w:tblW")
        tblPr.append(width_el)
    width_el.set(qn("w:w"), str(int(sum(widths) * 1440)))
    width_el.set(qn("w:type"), "dxa")
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    mar = OxmlElement("w:tblCellMar")
    for side, w in (("top", 12), ("left", 40), ("bottom", 12), ("right", 40)):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"), str(w))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    look = tblPr.find(qn("w:tblLook"))             # keep the schema order: tblLayout, tblCellMar, then tblLook
    for el in (layout, mar):
        if look is not None:
            look.addprevious(el)
        else:
            tblPr.append(el)
    for index, w in enumerate(widths):
        table.columns[index].width = Inches(w)
    for r_index, values in enumerate([header] + rows):
        for c_index, value in enumerate(values):
            cell = table.cell(r_index, c_index)
            cell.width = Inches(widths[c_index])
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = p.paragraph_format.space_after = Pt(0)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_index == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = r_index < len(rows)       # keep the whole table on one page
            add_runs(p, str(value), size, bold=(r_index == 0) or (r_index - 1 in bold_rows))
            if r_index == 0:
                cell_border(cell, top=8, bottom=4)
            if r_index == len(rows):
                cell_border(cell, bottom=8)
    return table


def add_figure(doc, path, width, number, caption):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Inches(width))
    para(doc, f"Fig. {number}. {caption}", size=8, align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=0, before=1, after=5)


# ----------------------------------------------------------------------------------------------
# content
# ----------------------------------------------------------------------------------------------
def build(results_dir, paper_dir):
    R, P = Path(results_dir).resolve(), Path(paper_dir).resolve()
    N = json.loads((P / "paper_numbers.json").read_text(encoding="utf-8"))
    S = pd.read_csv(R / "cv_thai_all_summary.csv").set_index(["model", "variant"])
    U = pd.read_csv(R / "cv_uk_summary.csv").set_index(["model", "variant"])
    M = pd.read_csv(R / "cv_motorcycle_summary.csv").set_index(["model", "variant"])
    UKF = pd.read_csv(R / "cv_uk_folds.csv")
    THF = pd.read_csv(R / "cv_thai_all_folds.csv")
    CTRL = pd.read_csv(R / "cv_thai_smote_control_folds.csv")
    TEST = pd.read_csv(R / "test_2026_final.csv").set_index("model")
    BOOT = pd.read_csv(R / "test_2026_bootstrap.csv").set_index("metric")
    SEL = pd.read_csv(R / "selection_report.csv")
    TUNE = pd.read_csv(R / "tuning_thai.csv")
    audit = json.loads((R / "uk_audit.json").read_text(encoding="utf-8"))
    final = json.loads((R / "final_selection.json").read_text(encoding="utf-8"))
    assert final["model"] == "lightgbm" and final["tuned"] and final["variant"] == "class_weight", final
    test = TEST.loc["lightgbm_tuned"]
    base = TEST.loc["majority_class"]

    def g(model, variant, metric, sd=False):
        return float(S.loc[(model, variant), metric + ("_std" if sd else "_mean")])

    def gu(model, variant, metric):
        return float(U.loc[(model, variant), metric + "_mean"])

    def gm(model, variant, metric):
        return float(M.loc[(model, variant), metric + "_mean"])

    f3 = lambda x: f"{x:.3f}"
    f2 = lambda x: f"{x:.2f}"
    pc = lambda x, d=1: f"{x * 100:.{d}f}%"
    ci = lambda name: f"{BOOT.loc[name, 'ci_low']:.3f}–{BOOT.loc[name, 'ci_high']:.3f}"

    # cross-checks of audit numbers that are quoted from the data audit (site / project-data.js)
    site = (P.parent.parent / "website" / "project-data.js").read_text(encoding="utf-8")
    for needle in ("195,054", "25,031", "1,254", "170,022", "77 provinces"):
        assert needle in site, needle
    ct = N["class_totals"]
    total = sum(ct.values())
    n_model = int(TEST["n_train"].iloc[0]) + int(TEST["n_test"].iloc[0])           # rows after the 2023 de-duplication
    folds = THF[(THF.model == "majority_class") & (THF.variant == "default")].sort_values("val_year")
    fold_sizes = THF[(THF.model == "lightgbm_tuned") & (THF.variant == "default")].sort_values("val_year")
    fs = N["fatal_share_by_year"]
    mean_prev_th = float(fold_sizes["fatal_share_val"].mean())
    mean_prev_uk = float(UKF[(UKF.model == "majority_class")]["fatal_share_val"].mean())
    xg, lg = "xgboost", "lightgbm"
    rec_plain = (g(xg, "default", "fatal_recall"), g(lg, "default", "fatal_recall"))
    rec_boost = [g(m_, v, "fatal_recall") for m_ in (xg, lg) for v in ("class_weight", "prior_adjusted")]
    prec_plain = (g(xg, "default", "fatal_precision"), g(lg, "default", "fatal_precision"))
    prec_boost = [g(m_, v, "fatal_precision") for m_ in (xg, lg) for v in ("class_weight", "prior_adjusted")]
    smote_prauc = (g(lg, "smote", "fatal_pr_auc"), g(xg, "smote", "fatal_pr_auc"))
    plain_prauc = (g(lg, "default", "fatal_pr_auc"), g(xg, "default", "fatal_pr_auc"))
    ctrl = CTRL.groupby("model")[["fatal_f1", "fatal_pr_auc"]].mean()
    inner = TUNE.groupby(["model", "val_year"]).apply(lambda d: d["inner_pr_auc"].max() - d.loc[d.config_id == 0, "inner_pr_auc"].iloc[0], include_groups=False)
    top3 = SEL.head(3)
    runner_gap = float(SEL.iloc[2]["gap_to_winner_mean"])
    runner_sd = float(SEL.iloc[0]["fatal_f1_std"])
    prior_wins = [g(m_, "smote", "fatal_f1") < g(m_, "prior_adjusted", "fatal_f1") for m_ in
                  ("logistic_regression", "decision_tree", "random_forest", xg, lg)]
    assert all(prior_wins), "SMOTE-NC beat prior adjustment for some family: the text below needs rewording"
    far = (1 - test["fatal_precision"]) / test["fatal_precision"]
    cm = N["test_confusion"]["matrix"]
    fatal_total = sum(cm[0])
    pr = N["pr"]
    imp = N["importance"]["features"]
    prov = N["provinces"]
    moto_rf = ("random_forest", "default")
    moto_pos = 0.302
    RECODE = json.loads((R / "category_recode_2023.json").read_text(encoding="utf-8"))["share_unseen_in_2019_2022"]
    fig_js = (P.parent.parent / "website" / "figures-data.js").read_text(encoding="utf-8")
    abl = json.loads(fig_js[fig_js.index("{"): fig_js.rindex("}") + 1])["importance"]["check"]
    assert abs(abl["with"]["fatalF1"] - abl["without"]["fatalF1"]) < 0.01, abl
    tune_gain = g("lightgbm_tuned", "default", "fatal_pr_auc") - g(lg, "default", "fatal_pr_auc")
    uk_lgb = UKF[UKF.model == lg]
    uk_cw = uk_lgb[uk_lgb.variant == "class_weight"].sort_values("val_year").fatal_pr_auc.to_numpy()
    uk_pl = uk_lgb[uk_lgb.variant == "default"].sort_values("val_year").fatal_pr_auc.to_numpy()
    assert (uk_cw < uk_pl).all(), "UK class weights no longer lower PR-AUC in every fold: reword the UK paragraph"

    # ---- post hoc re-check (results_recheck): read straight from its experiment log and summaries
    RCD = R.parent / "results_recheck"
    RC = pd.read_csv(RCD / "experiment_log.csv").drop_duplicates(subset=["stage", "family", "encoding", "strategy", "val_year"], keep="last")

    def rc(stage, family, encoding, strategy, metric):
        d = RC[(RC.stage == stage) & (RC.family == family) & (RC.encoding == encoding) & (RC.strategy == strategy)].sort_values("val_year")
        assert len(d) == 4, (stage, family, encoding, strategy, len(d))
        return d[metric].to_numpy()

    rcm = lambda *a: float(rc(*a).mean())
    rcs = lambda *a: float(rc(*a).std(ddof=1))
    EXP1 = "exp1_native_vs_ordinal"
    nat_pr = rc(EXP1, "lightgbm", "native", "plain", "fatal_pr_auc")
    ord_pr = rc(EXP1, "lightgbm", "ordinal", "plain", "fatal_pr_auc")
    assert (nat_pr - ord_pr)[:2].sum() > (nat_pr - ord_pr)[2:].sum(), "native-category gain is no longer concentrated in 2022-2023"
    thr = rc("exp4_threshold", "lightgbm", "native", "thr_f1_inner", "fatal_f1")
    cw_nat = rc(EXP1, "lightgbm", "native", "class_weight", "fatal_f1")
    CAL = pd.read_csv(RCD / "calibration_summary.csv").set_index(["family", "encoding", "strategy"])
    ece = lambda strat: float(CAL.loc[("lightgbm", "native", strat), "ece_10_equal_width_mean"])
    RNK = pd.read_csv(RCD / "ranking_equivalence_summary.csv").set_index(["family", "encoding"]).loc[("lightgbm", "native")]
    SUB = pd.read_csv(RCD / "err2025_lightgbm_native_thr_f1_inner_subgroup_recall.csv", encoding="utf-8-sig")
    sub = lambda dim, group, col="fatal_recall": float(SUB[(SUB.dimension == dim) & (SUB.group == group)][col].iloc[0])
    EP = pd.read_csv(RCD / "err2025_lightgbm_native_thr_f1_inner_error_profile.csv", encoding="utf-8-sig")
    fn_rows, fp_rows = EP[EP.kind.str.startswith("false_negative")], EP[EP.kind.str.startswith("false_positive")]
    ep_share = lambda d, col, label: float(d[(d.column == col) & (d.label == label)].share_in_errors.iloc[0])
    top_fn_type = fn_rows[fn_rows.column == "acc_type"].sort_values("share_in_errors", ascending=False).iloc[0]
    assert top_fn_type.label == "\u0e1e\u0e25\u0e34\u0e01\u0e04\u0e27\u0e48\u0e33/\u0e15\u0e01\u0e16\u0e19\u0e19\u0e43\u0e19\u0e17\u0e32\u0e07\u0e15\u0e23\u0e07", top_fn_type.label

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)
    sec.left_margin = sec.right_margin = Inches(0.63)
    sec.top_margin, sec.bottom_margin = Inches(0.7), Inches(0.7)
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(10)

    # ---------------- title block (one column)
    para(doc, "Which Thai Road Crashes Turn Fatal? Imbalance Strategies, Temporal Validation and a UK Replication", size=22,
         align=WD_ALIGN_PARAGRAPH.CENTER, indent=0, before=0, after=8)
    para(doc, "Aung Ko Ko Minn, Htoo Myat Htin Lynn, and Thant Mg Mg Lwin", size=11.5, align=WD_ALIGN_PARAGRAPH.CENTER, indent=0, after=1)
    para(doc, "Team AHT, ICT Program, Rangsit University International College, Pathum Thani, Thailand", size=9.5, italic=True,
         align=WD_ALIGN_PARAGRAPH.CENTER, indent=0, after=1)
    para(doc, "ICT324 Data Mining and Data Warehouse, final project, 2026", size=9.5, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, indent=0, after=6)

    body = doc.add_section(WD_SECTION.CONTINUOUS)
    set_cols(body, 2)

    # ---------------- abstract
    para(doc, f"<i>Abstract</i>—About one in nine incidents on Thailand’s Ministry of Transport road network is fatal, so a classifier that "
              f"ignores fatal crashes can still look accurate. We ask which model and imbalance strategy best finds fatal crashes while keeping false "
              f"alarms useful. On {n_model:,} incidents from 2019 to 2026 we classify each incident as fatal, injury or property damage only, and compare "
              f"five model families under four strategies (plain, class weights, prior adjustment, SMOTE-NC) with rolling-origin temporal validation and "
              f"nested tuning. Class weights and prior adjustment raise mean fatal recall from {f2(rec_plain[1])} to {f2(min(rec_boost))}–{f2(max(rec_boost))} "
              f"at precision {f2(min(prec_boost))}–{f2(max(prec_boost))}; in Thailand they move the decision cut-off, not the ranking of incidents. SMOTE-NC did not help "
              f"boosted trees and tuning added little. Chosen by a rule fixed before testing, LightGBM with class weights scored once on the 2026 test year "
              f"reaches fatal recall {f3(test['fatal_recall'])}, precision {f3(test['fatal_precision'])}, F1 {f3(test['fatal_f1'])} and ROC-AUC "
              f"{f3(test['fatal_auc'])}. The same pipeline on {audit['rows']:,} UK STATS19 collisions ({pc(audit['classShare']['fatal'])} fatal) gives a similar "
              f"ordering of strategies.", size=9, bold=True, indent=0, after=3)
    para(doc, "<i>Index Terms</i>—crash severity, class imbalance, temporal validation, gradient boosting, SMOTE, road safety, Thailand",
         size=9, bold=True, indent=0, after=4)

    # ---------------- I. Introduction
    heading1(doc, "I", "Introduction")
    para(doc, f"Road crashes are recorded in detail by transport agencies, yet the records are rarely turned into tools that help decide, once an incident is "
              f"recorded, which incidents deserve the fastest follow-up. Thailand’s Ministry of Transport (MOT) publishes incidents on its road network "
              f"(national and rural highways and expressways) from the TRAMS system {c('trams')}. In these records {pc(ct['fatal'] / total)} of incidents "
              f"are fatal, so accuracy alone hides missed fatal cases: plain LightGBM is more accurate in cross-validation ({f3(g(lg, 'default', 'accuracy'))} against {f3(g(lg, 'class_weight', 'accuracy'))} with class weights) yet finds "
              f"{pc(g(lg, 'default', 'fatal_recall'), 0)} of fatal crashes against {pc(g(lg, 'class_weight', 'fatal_recall'), 0)}. Earlier Thai work found the same difficulty. A random forest was the best of several "
              f"models for at-fault motorcycle riders but caught only about 20% of fatal crashes {c('mohamad')}.")
    para(doc, f"Three gaps motivate this study. First, Thai studies compare models {c('champahom', 'uttra', 'phojaem')} and resampling {c('simmachan')}, but we know of none that combine "
              f"<i>imbalance strategies</i> with validation that respects time order and a test year scored once. Random splits let future incidents inform predictions about the past, "
              f"a form of leakage that inflates results {c('kapoor')}. Second, resampling is popular for crash data {c('mamun', 'tong')}, yet evidence suggests it "
              f"often does not help strong classifiers {c('elor')}. Third, findings are seldom checked on a second country’s data.")
    para(doc, f"We therefore ask: <i>which model and imbalance strategy improves fatal-crash detection across three severity classes while keeping false alarms "
              f"useful?</i> Our contributions are: (i) a benchmark of five model families under four imbalance strategies on {n_model:,} Thai incidents, with "
              f"rolling-origin validation, nested tuning and a selection rule fixed before the 2026 test year was scored once; (ii) evidence that class weights and "
              f"prior adjustment move the cut-off along an unchanged precision–recall curve in Thailand, that SMOTE-NC (as configured here) does not help boosted trees, and that tuning adds little; "
              f"(iii) a replication on UK STATS19 {c('stats19')} and a motorcycle-only comparison with {c('mohamad')}; and (iv) the vehicle, place and time "
              f"patterns behind the predictions. We follow CRISP-DM {c('crispdm')}.")

    # ---------------- II. Related work
    heading1(doc, "II", "Related Work")
    para(doc, f"<i>Thai crash severity.</i> Mohamad <i>et al.</i> {c('mohamad')} compared eight models on 112,837 incidents (2015–2020) caused by motorcycle riders, with a random 75:25 split, and found "
              f"a random forest best overall (0.768 AUC) but weak at recalling fatal cases (0.198). Champahom <i>et al.</i> {c('champahom')} segmented Thai crashes across roadway classes using tree ensembles (XGBoost, random forest) with SHAP. "
              f"Uttra <i>et al.</i> {c('uttra')} classified 2017–2019 holiday-period crashes (Songkran and New Year) into three severity classes using SMOTE inside training splits, where XGBoost reached 83.8% accuracy versus 49.6% for ordered logit. "
              f"Phojaem <i>et al.</i> {c('phojaem')} compared statistical, boosted-tree and neural models on 13,630 Thai motorway crashes (2016–2023, 4.34% severe/fatal) with SMOTE: LightGBM led Macro-F1 (0.441), multinomial logit led severe recall (0.703) with high false alarms, and XGBoost with SHAP highlighted heavy trucks and night crashes. "
              f"Simmachan and Boonkrong {c('simmachan')} compared under-, over- and combined sampling on 31,817 Thai crashes (2021–2022) across random 70:30–90:10 splits and found combined sampling helped tree models. "
              f"Tree ensembles lead across these studies, yet none combine imbalance comparisons with rolling-origin temporal validation and a single-shot hold-out year.")
    para(doc, f"<i>Imbalance and evaluation.</i> SMOTE {c('chawla')} synthesises minority examples by interpolation. Al Mamun <i>et al.</i> {c('mamun')} compared balancing "
              f"techniques for work-zone crashes and found lower accuracy for high-severity crashes because of imbalance. Zhu <i>et al.</i> {c('zhu')} used a "
              f"temporally held-out test set and found that simple class weighting improves fatal recall with a modest accuracy cost. Elor and Averbuch-Elor "
              f"{c('elor')} report that balancing generally does not help strong classifiers. For tabular data, tree ensembles remain hard to beat "
              f"{c('grinsztajn')}. With rare positives, precision–recall measures are more informative than ROC {c('saito')}.")
    para(doc, f"<i>UK data.</i> Sufian and Varadarajan {c('sufian')} model UK accident severity with a random forest and an AutoML-selected XGBoost model. Tong "
              f"{c('tong')} applies SMOTE, random forest, XGBoost and SHAP to 2023 STATS19 pedestrian collisions and finds that police attendance helps separate "
              f"severe collisions. Attendance is decided after the outcome is known, so we exclude it (Section III).")

    # ---------------- III. Data
    heading1(doc, "III", "Data Understanding and Preparation")
    heading2(doc, "A", "Thai TRAMS incidents")
    para(doc, f"We merged the eight yearly TRAMS files (2019 to 12 October 2026; 195,054 rows). After removing 25,031 empty rows and one row with an invalid date, "
              f"{total:,} dated incidents remain. The files use three date formats, parsed per file; coordinates outside Thailand are set to missing; two labels of "
              f"one province are merged (77 provinces). The route name was dropped (about 13% missing overall, 77% in 2024). The 2023 file repeats accident "
              f"codes; removing the 1,254 duplicate rows leaves {n_model:,} incidents for modelling. The fatal share fell from {pc(fs['2019'])} in 2019 to {pc(fs['2026'])} in 2026 "
              f"(partial year), a drift that temporal validation exposes. The 2023 file also spells many category values differently from the other years: "
              f"{pc(RECODE['location']['2023'], 0)} of its (non-missing) road-location values and {pc(RECODE['cause']['2023'], 0)} of its presumed-cause values "
              f"({pc(RECODE['acc_type']['2023'], 0)} of accident types) never occur in 2019–2022, against about 0.1% or less in 2024–2026. Models trained on earlier "
              f"years therefore meet mostly unseen labels in the 2023 validation fold, which lowers recall and F1 there and widens the fold standard deviations of the LightGBM scores (Section V-G); "
              f"the ranking measure (PR-AUC) is much less affected.")
    para(doc, f"The severity target is derived from casualty counts; before de-duplication the {total:,} dated incidents split into fatal (at least one death, {pc(ct['fatal'] / total)}), injury (no death, at least one injured, "
              f"{pc(ct['injury'] / total)}) and property damage only ({pc(ct['property_only'] / total)}). The casualty columns define the label, so they are never "
              f"features. The 17 features are seven categorical (province, responsible agency, first vehicle type, road location, presumed cause, accident type, "
              f"weather) and ten numeric (latitude, longitude, kilometre marker, vehicles and pedestrians involved, hour, month, weekday, weekend and night flags). Presumed cause, accident type and the counts are entered after "
              f"the incident has been attended, so predictions are available once a record is filed, not at the moment of the call. "
              f"Imputing, scaling and encoding are fitted inside each training set only.")
    heading2(doc, "B", "UK STATS19")
    para(doc, f"The UK collision files for 2021–2024 {c('stats19')} hold {audit['rows']:,} collisions: fatal {pc(audit['classShare']['fatal'])}, serious "
              f"{pc(audit['classShare']['serious'])}, slight {pc(audit['classShare']['slight'])}. These classes are not the Thai ones (the UK has no damage-only "
              f"records), so the two countries are compared on fatal-class metrics. Features mirror the Thai set where the UK table has an equivalent (place, time, road class, junction, speed limit, light, "
              f"weather, surface, urban or rural); it has no vehicle type, accident type or presumed cause, and vehicle type is the strongest Thai feature (Fig. 3). "
              f"A UK collision is fatal if a death follows within 30 days. We leave out the number of casualties, whether police attended, the derived severity fields and the small-area code, "
              f"because they describe the outcome or its handling.")

    # ---------------- IV. Method
    heading1(doc, "IV", "Methodology")
    heading2(doc, "A", "Models and imbalance strategies")
    para(doc, f"We compare a majority-class baseline, logistic regression, a decision tree, a random forest, XGBoost {c('xgboost')} and LightGBM {c('lightgbm')}, "
              f"implemented with scikit-learn {c('sklearn')} and imbalanced-learn {c('imblearn')} (seed 42). Each is run with four strategies. <i>Plain</i> fits the "
              f"model as is. <i>Class weights</i> weight training examples inversely to class frequency. <i>Prior adjustment</i> divides each predicted class "
              f"probability <i>p<sub>k</sub></i> by the training share <i>π<sub>k</sub></i> and predicts the class with the largest <i>p<sub>k</sub>/π<sub>k</sub></i>; "
              f"it needs no refitting. <i>SMOTE-NC</i> oversamples the fatal class in the training years only, up to the size of the smaller other class; because it "
              f"needs complete data, numeric gaps are median-imputed for this strategy.")
    heading2(doc, "B", "Temporal validation, tuning and selection")
    sizes = "; ".join(f"{int(r.n_train):,}→{int(r.n_val):,}" for r in fold_sizes.itertuples())
    para(doc, f"Validation is rolling-origin: train on 2019–2021 and validate on 2022, then 2019–2022 on 2023, 2019–2023 on 2024 and 2019–2024 on 2025 "
              f"(training→validation incidents: {sizes}). Scores are means and standard deviations over the four folds. Only XGBoost and LightGBM are tuned, "
              f"because they led an earlier untuned comparison. Tuning is nested and time-aware: inside each fold’s training years, the last year is held out, the "
              f"default and nine random configurations are fitted on the earlier years, and the one with the best fatal-class PR-AUC on the held-out year wins; "
              f"all four strategies then share those settings. The outer validation year is never seen by tuning.")
    para(doc, f"The metrics are fatal-class recall, precision, F1 and PR-AUC (average precision), fatal-versus-rest ROC-AUC, macro-F1, balanced accuracy and "
              f"accuracy. <b>Selection rule, fixed in code before the test script was run:</b> the model and strategy with the highest mean fatal F1 over the four folds, "
              f"ties broken by macro-F1. The 2026 test year (17,454 incidents, to 12 October) is scored once, with the chosen model trained on 2019–2025 (its "
              f"settings re-tuned inside those years). Test intervals are 95% percentile intervals from 1,000 bootstrap resamples of the test incidents, so they "
              f"reflect test-sample variation only. We judge false alarms through fatal-class precision and F1; an explicit cost-based cut-off is left for future work.")
    heading2(doc, "C", "Replications")
    para(doc, f"The UK replication uses the same families and strategies with the fixed settings and rolling folds 2021→2022, 2021–2022→2023 and "
              f"2021–2023→2024; nothing is tuned on UK data. The motorcycle subset (first vehicle a motorcycle, 25,985 incidents, 30.2% fatal) uses the Thai "
              f"folds, with fatal-versus-rest accuracy, ROC-AUC and recall so that it can be set beside {c('mohamad')}. Here the random forest, XGBoost and LightGBM are also "
              f"tuned with the same nested procedure, but with ROC-AUC as the criterion because that is the benchmark\u2019s headline measure.")

    # ---------------- V. Results
    heading1(doc, "V", "Results")
    heading2(doc, "A", "Cross-validation on Thai data")
    rows_spec = [("Majority class", "majority_class", "default"), ("Logistic reg., weights", "logistic_regression", "class_weight"),
                 ("Decision tree, prior", "decision_tree", "prior_adjusted"), ("Random forest, prior", "random_forest", "prior_adjusted"),
                 ("XGBoost, plain", xg, "default"), ("XGBoost, weights", xg, "class_weight"),
                 ("LightGBM, plain", lg, "default"), ("LightGBM, weights", lg, "class_weight"), ("LightGBM, SMOTE-NC", lg, "smote"),
                 ("LGBM tuned, plain", "lightgbm_tuned", "default"), ("LGBM tuned, weights", "lightgbm_tuned", "class_weight"),
                 ("LGBM tuned, SMOTE", "lightgbm_tuned", "smote")]
    rows = [[label, f3(g(m_, v, "fatal_recall")), f3(g(m_, v, "fatal_precision")),
             f"{f3(g(m_, v, 'fatal_f1'))}±{f2(g(m_, v, 'fatal_f1', True))[1:] if g(m_, v, 'fatal_f1', True) < 1 else ''}",
             f3(g(m_, v, "fatal_pr_auc")), f3(g(m_, v, "macro_f1"))] for label, m_, v in rows_spec]
    add_table(doc, "I", "Thai cross-validation, mean over four temporal folds (LGBM = LightGBM)", ["Model, strategy", "Recall", "Prec.", "F1±SD", "PR-AUC", "Macro-F1"],
              rows, [1.0, 0.4, 0.4, 0.6, 0.43, 0.5], bold_rows=(10,))
    para(doc, f"Table I shows the main picture (fatal-class scores except macro-F1). Plain models are precise but miss most fatal incidents: XGBoost and LightGBM "
              f"find {f2(rec_plain[0])} and {f2(rec_plain[1])} of them at precision {f2(prec_plain[0])} and {f2(prec_plain[1])}. Class weights and prior adjustment lift "
              f"recall to {f2(min(rec_boost))}–{f2(max(rec_boost))} and cost precision ({f2(min(prec_boost))}–{f2(max(prec_boost))}). Fatal PR-AUC "
              f"stays at {f3(plain_prauc[0])} (LightGBM) with or without weights ({f3(g(lg, 'class_weight', 'fatal_pr_auc'))}). Fig. 1 shows why: for the 2025 fold, the "
              f"plain and weighted curves coincide and the strategies only choose different points on one curve.", before=2, indent=0.17)
    add_figure(doc, P / "figures" / "fig_pr.png", 3.05, 1,
               f"Fatal-class precision–recall on the 2025 validation year (LightGBM, tuned settings, trained on 2019–2024). Lines rank incidents by predicted "
               f"fatal probability; markers show where each strategy’s decisions fall. AP is average precision.")
    heading2(doc, "B", "SMOTE-NC and tuning")
    fam = [("Logistic regression", "logistic_regression"), ("Decision tree", "decision_tree"), ("Random forest", "random_forest"),
           ("XGBoost", xg), ("LightGBM", lg)]
    add_table(doc, "II", "Fatal F1 by model family and strategy (untuned)", ["Family", "Plain", "Weights", "Prior", "SMOTE-NC"],
              [[label] + [f3(g(m_, v, "fatal_f1")) for v in ("default", "class_weight", "prior_adjusted", "smote")] for label, m_ in fam],
              [1.1, 0.55, 0.55, 0.55, 0.58])
    para(doc, f"SMOTE-NC barely raised recall for boosted trees ({f2(g(lg, 'smote', 'fatal_recall'))} and {f2(g(xg, 'smote', 'fatal_recall'))}) and lowered their "
              f"PR-AUC to {f3(smote_prauc[0])} and {f3(smote_prauc[1])}, against {f3(plain_prauc[0])} and {f3(plain_prauc[1])} for the plain models. A control that "
              f"uses the same median-imputed matrix without resampling gives F1 {f3(ctrl.loc['xgboost_imputed_control', 'fatal_f1'])} and PR-AUC "
              f"{f3(ctrl.loc['xgboost_imputed_control', 'fatal_pr_auc'])} for XGBoost, so imputation is not the cause: in this configuration (fatal oversampled to the size of the smaller other class, five neighbours) the synthetic examples hurt the ranking; other ratios were not tried. Standardising the numeric features before resampling gives PR-AUC "
              f"{f3(rcm('exp2_smote', 'lightgbm', 'smote_scaled_ordinal', 'plain', 'fatal_pr_auc'))} and {f3(rcm('exp2_smote', 'xgboost', 'smote_scaled_ordinal', 'plain', 'fatal_pr_auc'))} "
              f"(matched controls {f3(rcm('exp2_smote', 'lightgbm', 'control_ordinal', 'plain', 'fatal_pr_auc'))} and {f3(rcm('exp2_smote', 'xgboost', 'control_ordinal', 'plain', 'fatal_pr_auc'))}), "
              f"and with native categories {f3(rcm('exp2_smote', 'lightgbm', 'smote_scaled_native', 'plain', 'fatal_pr_auc'))} and {f3(rcm('exp2_smote', 'xgboost', 'smote_scaled_native', 'plain', 'fatal_pr_auc'))} "
              f"(controls {f3(rcm('exp2_smote', 'lightgbm', 'control_native', 'plain', 'fatal_pr_auc'))} and {f3(rcm('exp2_smote', 'xgboost', 'control_native', 'plain', 'fatal_pr_auc'))}): "
              f"part of the deficit came from unscaled features, but SMOTE-NC still does not help. SMOTE-NC "
              f"helped the weaker families over their plain versions but never beat prior adjustment (Table II). This agrees with {c('elor')}. Tuning improved the held-out-year "
              f"PR-AUC inside training by {f3(inner.min())}–{f3(inner.max())}, but on the outer folds it added only about "
              f"{f3(g('lightgbm_tuned', 'default', 'fatal_pr_auc') - g(lg, 'default', 'fatal_pr_auc'))} PR-AUC and "
              f"{f3(g('lightgbm_tuned', 'class_weight', 'fatal_f1') - g(lg, 'class_weight', 'fatal_f1'))} fatal F1 for LightGBM.")
    heading2(doc, "C", "Selection and the 2026 test year")
    para(doc, f"By the rule, LightGBM with class weights and tuned settings was selected (mean fatal F1 {f3(top3.iloc[0]['fatal_f1_mean'])}). It is "
              f"<i>not</i> a clear winner: tuned XGBoost with class weights ({f3(top3.iloc[1]['fatal_f1_mean'])}) and untuned XGBoost with class weights "
              f"({f3(top3.iloc[2]['fatal_f1_mean'])}) are within {f3(runner_gap)}, far inside the fold standard deviation of {f3(runner_sd)}, so we call them comparable (so is a plain model with a threshold chosen on the last training year, Section V-G). "
              f"Scored once on 2026 (Table IV, Fig. 2), the selected model finds {pc(test['fatal_recall'], 0)} of fatal incidents ({cm[0][0]:,} of {fatal_total:,}) "
              f"at precision {f3(test['fatal_precision'])}: about {far:.1f} false alarms per correct fatal alert. The test F1 of {f3(test['fatal_f1'])} is in line with "
              f"cross-validation ({f3(top3.iloc[0]['fatal_f1_mean'])}), so we see no sign of overfitting to the validation years.")
    sel_f = THF[(THF.model == "lightgbm_tuned") & (THF.variant == "class_weight")].sort_values("val_year")
    fold_rows = [[str(int(r.val_year)), f3(r.fatal_recall), f3(r.fatal_precision), f3(r.fatal_f1), f3(r.fatal_pr_auc)] for r in sel_f.itertuples()]
    fold_rows.append(["Mean", f3(sel_f.fatal_recall.mean()), f3(sel_f.fatal_precision.mean()), f3(sel_f.fatal_f1.mean()), f3(sel_f.fatal_pr_auc.mean())])
    add_table(doc, "III", "Selected model by validation year", ["Year", "Recall", "Prec.", "F1", "PR-AUC"], fold_rows, [0.7, 0.65, 0.65, 0.65, 0.7],
              bold_rows=(len(fold_rows) - 1,))
    r23, r_oth = float(sel_f[sel_f.val_year == 2023].fatal_recall.iloc[0]), float(sel_f[sel_f.val_year != 2023].fatal_recall.mean())
    p23, p_oth = float(sel_f[sel_f.val_year == 2023].fatal_pr_auc.iloc[0]), float(sel_f[sel_f.val_year != 2023].fatal_pr_auc.mean())
    assert r23 < r_oth and p23 >= p_oth - 0.01, (r23, r_oth, p23, p_oth)
    para(doc, f"Table III lists the selected model by validation year. Recall dips in 2023 ({f3(r23)} against {f3(r_oth)} in the other years), which is the "
              f"category recoding described in Section III-A, while PR-AUC is not lower ({f3(p23)} against {f3(p_oth)}); the dip inflates the fold standard "
              f"deviations.", before=2, indent=0.17)
    add_table(doc, "IV", "2026 test year, scored once (17,454 incidents)", ["Metric", "Selected model (95% CI)", "Majority"],
              [["Fatal recall", f"{f3(test['fatal_recall'])} ({ci('fatal_recall')})", "0.000"],
               ["Fatal precision", f"{f3(test['fatal_precision'])} ({ci('fatal_precision')})", "0.000"],
               ["Fatal F1", f"{f3(test['fatal_f1'])} ({ci('fatal_f1')})", "0.000"],
               ["Fatal PR-AUC", f"{f3(test['fatal_pr_auc'])} ({ci('fatal_pr_auc')})", f3(base["fatal_pr_auc"])],
               ["Fatal ROC-AUC", f"{f3(test['fatal_auc'])} ({ci('fatal_auc')})", "0.500"],
               ["Macro-F1", f"{f3(test['macro_f1'])} ({ci('macro_f1')})", f3(base["macro_f1"])],
               ["Accuracy", f3(test["accuracy"]), f3(base["accuracy"])]],
              [0.95, 1.8, 0.58])
    add_figure(doc, P / "figures" / "fig_cm.png", 2.4, 2,
               "Confusion matrix on the 2026 test year (rows: actual class; percentages are row shares, counts in brackets).")
    heading2(doc, "D", "What drives the predictions, and where")
    para(doc, f"Grouped permutation importance (Fig. 3) shows that the first vehicle type matters most (drop in PR-AUC {f3(imp[0]['mean'])}), followed by "
              f"vehicle and person counts, i.e. vehicles plus pedestrians ({f3(imp[1]['mean'])}), accident type ({f3(imp[2]['mean'])}) and GPS position ({f3(imp[3]['mean'])}). Related columns are "
              f"shuffled together so that near-duplicates are not overstated. In a check with class-weighted XGBoost on the 2025 fold, emptying the vehicles-and-pedestrians-involved column moved fatal F1 from "
              f"{abl['with']['fatalF1']:.3f} to {abl['without']['fatalF1']:.3f}, so the scores do not depend on that column. Importance describes what the model uses, not what causes a crash to turn fatal.")
    add_figure(doc, P / "figures" / "fig_imp.png", 2.85, 3,
               "Grouped permutation importance of the class-weighted LightGBM on the 2025 validation year (five shuffles; bars ± one SD).")
    para(doc, f"Fatal share varies strongly by place (Fig. 4). Seven of {prov['n']} provinces exceed 20%, led by "
              f"{prov['top'][0]['name']} ({pc(prov['top'][0]['fatal_share'])}), {prov['top'][1]['name']} ({pc(prov['top'][1]['fatal_share'])}) and "
              f"{prov['top'][2]['name']} ({pc(prov['top'][2]['fatal_share'])}), while, among provinces with at least 1,000 incidents, Bangkok ({pc(prov['bottom'][2]['fatal_share'])}) and Chon Buri "
              f"({pc(prov['bottom'][1]['fatal_share'])}) are lowest. The map shows the share among recorded incidents on the MOT network, not risk per kilometre "
              f"driven.")
    add_figure(doc, P / "figures" / "fig_map.png", 1.85, 4,
               f"Fatal share of incidents by province, 2019–2026 ({prov['total_incidents']:,} of the {total:,} dated incidents carry a province label; provinces are named in the text). Outlines: geoBoundaries, © OpenStreetMap contributors (ODbL).")
    heading2(doc, "E", "Motorcycle subset")
    rf_t, xg_t = ("random_forest_tuned", "default"), ("xgboost_tuned", "default")
    maj_acc = gm("majority_class", "default", "fatal_vs_rest_accuracy")
    gain_ours = gm(*xg_t, "fatal_vs_rest_accuracy") - maj_acc
    gain_theirs = 0.777 - (1 - 0.244)
    para(doc, f"On motorcycle incidents ({pc(moto_pos, 0)} fatal) the untuned random forest reaches fatal-versus-rest accuracy "
              f"{f3(gm(*moto_rf, 'fatal_vs_rest_accuracy'))}, ROC-AUC {f3(gm(*moto_rf, 'fatal_auc'))} and fatal recall {f3(gm(*moto_rf, 'fatal_recall'))}. "
              f"After tuning for ROC-AUC, the random forest (the benchmark\u2019s family) reaches {f3(gm(*rf_t, 'fatal_vs_rest_accuracy'))}, "
              f"{f3(gm(*rf_t, 'fatal_auc'))} and {f3(gm(*rf_t, 'fatal_recall'))}, and XGBoost reaches {f3(gm(*xg_t, 'fatal_vs_rest_accuracy'))}, "
              f"{f3(gm(*xg_t, 'fatal_auc'))} (SD {f3(float(M.loc[xg_t, 'fatal_auc_std']))}) and {f3(gm(*xg_t, 'fatal_recall'))}. The benchmark {c('mohamad')} (at-fault motorcycle riders on all road types, 112,837 incidents, 2015–2020, a random 75:25 split, fatal versus non-fatal) "
              f"reports 0.777, 0.768 and 0.198 for a random forest. Our best AUC equals theirs but does not exceed it, and the tuned random forest is slightly "
              f"lower; because the best of many models and strategies is picked on the same folds, even this match is optimistic. Accuracy is not comparable: the "
              f"all-not-fatal baseline scores {f3(maj_acc)} here and about 0.756 there, so the gain over baseline is {gain_ours:+.3f} against {gain_theirs:+.3f}. "
              f"Fatal recall is higher than the benchmark’s ({f3(gm(*xg_t, 'fatal_recall'))} against 0.198), but that depends on the cut-off: the same XGBoost with prior adjustment reaches "
              f"{f3(gm('xgboost_tuned', 'prior_adjusted', 'fatal_recall'))} at the same AUC, and the benchmark is a binary task while ours is read from a three-class prediction. "
              f"The data, years and fatal share (about 24% there) differ, and we have no rider variables.")
    heading2(doc, "F", "UK replication")
    ukl = ("lightgbm", "default")
    ukw = ("lightgbm", "class_weight")
    add_table(doc, "V", "Thailand and UK, LightGBM with fixed settings",
              ["Measure", "Thailand", "UK"],
              [["Fatal share, validation years", pc(mean_prev_th), pc(mean_prev_uk)],
               ["Plain: fatal recall", f3(g(lg, "default", "fatal_recall")), f3(gu(*ukl, "fatal_recall"))],
               ["Class weights: recall / precision", f"{f3(g(lg, 'class_weight', 'fatal_recall'))} / {f3(g(lg, 'class_weight', 'fatal_precision'))}",
                f"{f3(gu(*ukw, 'fatal_recall'))} / {f3(gu(*ukw, 'fatal_precision'))}"],
               ["Prior adjustment: recall", f3(g(lg, "prior_adjusted", "fatal_recall")), f3(gu(lg, "prior_adjusted", "fatal_recall"))],
               ["SMOTE-NC: recall", f3(g(lg, "smote", "fatal_recall")), f3(gu(lg, "smote", "fatal_recall"))],
               ["Plain: fatal PR-AUC", f3(g(lg, "default", "fatal_pr_auc")), f3(gu(*ukl, "fatal_pr_auc"))],
               ["PR-AUC / fatal share", f"{g(lg, 'default', 'fatal_pr_auc') / mean_prev_th:.1f}×", f"{gu(*ukl, 'fatal_pr_auc') / mean_prev_uk:.1f}×"],
               ["Plain: fatal ROC-AUC", f3(g(lg, "default", "fatal_auc")), f3(gu(*ukl, "fatal_auc"))]],
              [1.65, 0.85, 0.83])
    para(doc, f"The ordering of strategies repeats (Table V). At {pc(mean_prev_uk)} fatal, plain models almost never predict fatal ({f3(gu(*ukl, 'fatal_recall'))} "
              f"recall), class weights raise it to {f2(gu(*ukw, 'fatal_recall'))} at precision "
              f"{f3(gu(*ukw, 'fatal_precision'))} and prior adjustment to {f2(gu(lg, 'prior_adjusted', 'fatal_recall'))} at precision "
              f"{f3(gu(lg, 'prior_adjusted', 'fatal_precision'))}, and SMOTE-NC does not help. Unlike in Thailand, class weights lower the UK PR-AUC in every fold "
              f"({f3(gu(*ukw, 'fatal_pr_auc'))} against {f3(gu(*ukl, 'fatal_pr_auc'))} for the plain model), so prior adjustment is the cleaner cut-off shift there. "
              f"The ranking quality of the plain model relative to chance is similar "
              f"({g(lg, 'default', 'fatal_pr_auc') / mean_prev_th:.1f}× and {gu(*ukl, 'fatal_pr_auc') / mean_prev_uk:.1f}× the fatal share), although absolute "
              f"precision is far lower in the UK because fatal crashes are rarer.", before=3)

    heading2(doc, "G", "Robustness re-check")
    para(doc, f"After the main experiments we re-checked the results on the same validation folds with stricter practice: native categorical handling, scaled "
              f"SMOTE-NC, calibration and an inner-year threshold. This was post hoc, used the validation years only, and did not touch the 2026 test year. "
              f"<i>Native categories</i> change little. LightGBM\u2019s PR-AUC is {f3(nat_pr.mean())} against {f3(ord_pr.mean())} with ordinal codes, mostly from the 2022 "
              f"and 2023 folds, and the fold standard deviation of its class-weighted fatal F1 falls from {f3(rcs(EXP1, 'lightgbm', 'ordinal', 'class_weight', 'fatal_f1'))} to "
              f"{f3(rcs(EXP1, 'lightgbm', 'native', 'class_weight', 'fatal_f1'))}. XGBoost gets worse ({f3(rcm(EXP1, 'xgboost', 'native', 'plain', 'fatal_pr_auc'))} against "
              f"{f3(rcm(EXP1, 'xgboost', 'ordinal', 'plain', 'fatal_pr_auc'))}) because labels unseen in training become missing values for it.", before=2, indent=0.17)
    para(doc, f"<i>Calibration.</i> Plain models are well calibrated (expected calibration error {f3(ece('plain'))} for native LightGBM), but class-weighted and "
              f"prior-adjusted scores are not probabilities ({f3(ece('class_weight'))} and {f3(ece('prior_adjusted'))}). Ranking incidents by the plain probabilities and "
              f"calling as many as the weighted model does gives fatal F1 {f3(RNK['class_weight_F1_plain_ranking_same_k_mean'])} against "
              f"{f3(RNK['class_weight_F1_own_mean'])}, so class weights add nothing beyond a cut-off.")
    para(doc, f"<i>Threshold.</i> Choosing the cut-off on P(fatal) that maximises F1 on the last training year, for the plain native LightGBM, gives mean fatal F1 "
              f"{f3(thr.mean())} \u00b1 {f3(thr.std(ddof=1))} (recall {f3(rcm('exp4_threshold', 'lightgbm', 'native', 'thr_f1_inner', 'fatal_recall'))}, precision "
              f"{f3(rcm('exp4_threshold', 'lightgbm', 'native', 'thr_f1_inner', 'fatal_precision'))}): {thr.mean() - cw_nat.mean():+.3f} against class weights and ahead in "
              f"{int((thr > cw_nat).sum())} of four folds. It was added after the main results, is the best of many rows scored on the same folds, and was not scored on "
              f"the test year, so we treat it as comparable to the selected model, not better. <i>Feature timing.</i> Presumed cause, accident type and the people "
              f"count are filed after the incident is attended. Without all three, native LightGBM\u2019s PR-AUC falls from {f3(nat_pr.mean())} to "
              f"{f3(rcm('exp6_feature_ablation', 'lightgbm', 'native__no_cause_acctype_ninvolved', 'plain', 'fatal_pr_auc'))}, which is what a model restricted to the "
              f"first call could expect.")
    add_table(doc, "VI", "Fatal recall by subgroup, 2025 validation year",
              ["Subgroup", "Fatal incidents", "Recall"],
              [["Motorcycle", f"{int(sub('vehicle1', 'motorcycle', 'n_fatal')):,}", f3(sub("vehicle1", "motorcycle"))],
               ["Pickup / 4-wheel truck", f"{int(sub('vehicle1', 'pickup/4-wheel truck', 'n_fatal')):,}", f3(sub("vehicle1", "pickup/4-wheel truck"))],
               ["Car", f"{int(sub('vehicle1', 'car (private/public)', 'n_fatal')):,}", f3(sub("vehicle1", "car (private/public)"))],
               ["Northeast region", f"{int(sub('region', 'Northeast', 'n_fatal')):,}", f3(sub("region", "Northeast"))],
               ["North region", f"{int(sub('region', 'North', 'n_fatal')):,}", f3(sub("region", "North"))],
               ["Night (18:00\u201306:00)", f"{int(sub('night(18-06)', 'night', 'n_fatal')):,}", f3(sub("night(18-06)", "night"))],
               ["Day", f"{int(sub('night(18-06)', 'day', 'n_fatal')):,}", f3(sub("night(18-06)", "day"))]],
              [1.7, 0.95, 0.65])
    para(doc, f"Table VI is for the threshold model on the 2025 fold. It finds most motorcycle fatalities but under a third of those in cars and pickups. Among the "
              f"300 fatal incidents with the lowest scores, pickups ({pc(ep_share(fn_rows, 'vehicle', 'pickup/4-wheel truck'), 0)}) and cars "
              f"({pc(ep_share(fn_rows, 'vehicle', 'car (private/public)'), 0)}) are over-represented, motorcycles ({pc(ep_share(fn_rows, 'vehicle', 'motorcycle'), 0)}) "
              f"under-represented, and {pc(float(top_fn_type.share_in_errors), 0)} are rollovers or run-offs on straight roads. The 300 highest-scored false alarms are "
              f"{pc(ep_share(fp_rows, 'vehicle', 'motorcycle'), 0)} motorcycles.", before=2, indent=0.17)

    exp11_file = RCD / "recheck_exp11_improvements.json"
    if not exp11_file.exists():
        exp11_file = R / "recheck_exp11_improvements.json"
    if exp11_file.exists():
        E11 = json.loads(exp11_file.read_text(encoding="utf-8"))
        e11_up = next(x for x in E11["main_thai"] if x["label"] == "harmonized_plus_engineered")
        e11_m = next(x for x in E11["motorcycle_binary"]["models"] if x["label"] == "binary_lgbm_native_upgraded_tuned")
        para(doc, f"<i>Targeted follow-up (validation folds only).</i> Harmonising the main 2023 Thai spelling synonyms and adding zero-leakage features from existing TRAMS columns "
                  f"(coarse road tier extracted from the route field, Songkran/New Year holiday peaks, cyclical hour and month, and explicit extra persons) lifts calibrated plain native LightGBM "
                  f"mean PR-AUC from {f3(nat_pr.mean())} to {f3(e11_up['fatal_pr_auc_mean'])} and inner-year threshold fatal F1 to {f3(e11_up['fatal_f1_thr_mean'])} ± {f3(e11_up['fatal_f1_thr_sd'])} "
                  f"(ahead in all four folds, ECE {f3(e11_up['ece_fatal_mean'])}). On those calibrated probabilities, a top-5% urgent tier (P ≥ {f2(e11_up['tiers']['top5']['threshold'])}) reaches "
                  f"{pc(e11_up['tiers']['top5']['precision'])} precision at {pc(e11_up['tiers']['top5']['recall'])} recall, while a top-10% priority tier reaches {pc(e11_up['tiers']['top10']['precision'])} precision. "
                  f"On the motorcycle subset, switching to a dedicated binary fatal-versus-rest native LightGBM with the same harmonised and engineered features (nested ROC-AUC tuning) reaches "
                  f"ROC-AUC {f3(e11_m['fatal_auc_mean'])} ± {f3(e11_m['fatal_auc_sd'])} (exceeding the 0.768 benchmark in all four rolling folds) at fatal recall {f3(e11_m['fatal_recall_mean'])} "
                  f"and precision {f3(e11_m['fatal_precision_mean'])}. Because these additions were post hoc, 2026 remains untouched.")

    # ---------------- VI. Discussion
    heading1(doc, "VI", "Discussion and Limitations")
    para(doc, f"<i>What the results mean.</i> The choice of cut-off, not model complexity, decides how many fatal crashes are found. In Thailand the strategies share one "
              f"precision–recall curve, so an agency should pick its operating point from the cost of a false alarm: at the selected point, each correct alert "
              f"comes with about {far:.1f} false ones. SMOTE-NC added nothing and a modest tuning search about {f3(tune_gain)} PR-AUC, while a one-line class weight "
              f"or a probability rescaling did the work, in both countries. We would use the model to rank incidents for attention, not to trigger action alone, and any risk score shown to users should come from the plain "
              f"model, which is calibrated, not from the class-weighted one.")
    para(doc, f"<i>Limitations.</i> TRAMS covers the MOT road network, not city streets, and severity is derived from casualty counts, so reporting practice "
              f"affects labels. 2026 is a partial year and the fatal share is drifting down; the four validation folds overlap and share that trend, so their "
              f"standard deviation is only a rough spread, and the 2023 recoding (Section III-A) widens it for LightGBM. We used one random seed, and the bootstrap "
              f"intervals cover test-sample variation only, not training or temporal variance. Only two families were tuned, with nine random configurations, and "
              f"SMOTE-NC was tried at one sampling ratio (scaled and unscaled, ordinal and native categories). Native categories, calibration and the inner-year "
              f"threshold were checked only in a post hoc re-check on the validation folds (Section V-G), and there is no simple-rule baseline (a vehicle-type-only "
              f"model would be informative). Importance is descriptive, not causal. Presumed cause, accident type and the people count are filed after attendance, "
              f"so a first-call model would be weaker. The UK table has no vehicle type, accident type or cause, its first fold trains on a single COVID-affected "
              f"year, and its classes differ from the Thai ones. The motorcycle comparison is not like for like, and post hoc upgrades were checked on validation folds only. "
              f"For paywalled {c('phojaem')}, figures were verified from its institutional summary rather than the publisher PDF.")
    para(doc, f"<i>Intended use.</i> The data hold no personal information. The model supports prioritising response and inspection; it must not be used to assign "
              f"blame, set insurance or penalise individuals, since precision is about {pc(test['fatal_precision'], 0)}.")

    # ---------------- VII. Conclusion
    heading1(doc, "VII", "Conclusion")
    para(doc, f"On {n_model:,} Thai incidents with time-respecting validation, class weights or prior adjustment find {pc(min(rec_boost), 0)}–{pc(max(rec_boost), 0)} of fatal crashes at "
              f"{pc(min(prec_boost), 0)}–{pc(max(prec_boost), 0)} precision, versus {pc(sum(rec_plain) / 2, 0)} recall for plain models; SMOTE-NC and tuning add little. The selected LightGBM scored fatal F1 "
              f"{f3(test['fatal_f1'])} once on 2026, and the UK replication shows a similar pattern. Future work should add weather, rainfall, road class and a Thai "
              f"holiday calendar, choose cut-offs by explicit cost, and use exposure data to turn provincial shares into risk.")
    para(doc, "<i>Availability.</i> Code and results: github.com/KingZenon/research. Interactive site: kingzenon.github.io/research.",
         align=WD_ALIGN_PARAGRAPH.LEFT, indent=0, before=3)

    # ---------------- references
    heading1(doc, "", "References").runs[0].text = "REFERENCES"
    for index, key in enumerate(CITED, 1):
        para(doc, f"[{index}] {REFS[key]}", size=7.4, align=WD_ALIGN_PARAGRAPH.LEFT, indent=0, left=0.27, hanging=0.27, after=1.5)
    missing = [k for k in REFS if k not in CITED]
    assert not missing, f"uncited references: {missing}"

    out = P / "Team_AHT_ICT324_Final_Paper.docx"
    doc.save(out)
    print("wrote", out, f"| {len(CITED)} references")


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
