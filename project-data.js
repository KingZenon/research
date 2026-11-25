/*
 * Edit this file when the team supplies data, results, and final materials.
 * Keep unknown values as null. Do not enter estimated record counts or scores.
 * See README.md for the exact update steps and file locations.
 */
window.PROJECT_DATA = {
  lastUpdated: "24 November 2026",
  teamName: "Team AHT",

  // Short related-work items and cleaning steps (one line each).
  relatedWork: [
    {
      "cite": "Mohamad et al. (2025)",
      "text": "112,837 Thai at-fault motorcycle crashes (2015–2020, 75:25 random split). Random forest reached 0.768 AUC but caught only 19.8% of fatal crashes."
    },
    {
      "cite": "Champahom et al. (2024)",
      "text": "Thai crash severity segmented across roadway classes (highways, rural/urban) with tree ensembles (XGBoost, random forest) and SHAP."
    },
    {
      "cite": "Uttra et al. (2026)",
      "text": "Thai holiday crashes (2017–2019, Songkran and New Year) in three classes with SMOTE on training splits; XGBoost reached 83.8% accuracy vs. 49.6% for ordered logit (RPOLHM)."
    },
    {
      "cite": "Phojaem et al. (2026)",
      "text": "13,630 Thai motorway crashes (2016–2023, 4.34% severe/fatal) with SMOTE; LightGBM led Macro-F1 (0.441), MNL led severe recall (0.703) with high false alarms, and XGBoost+SHAP highlighted heavy trucks and night crashes."
    },
    {
      "cite": "Simmachan & Boonkrong (2025)",
      "text": "31,817 Thai crashes (2021–2022) across random 70/30–90/10 splits comparing RF, KNN and XGBoost under under-, over- and combined sampling."
    },
    {
      "cite": "Al Mamun et al. (2025)",
      "text": "Work-zone crashes under data imbalance: lower accuracy for high-severity crashes."
    },
    {
      "cite": "Zhu et al. (2025)",
      "text": "Temporal hold-out; simple class weighting improves fatal recall at a modest accuracy cost."
    },
    {
      "cite": "Elor & Averbuch-Elor (2022)",
      "text": "Balancing generally does not help strong classifiers (73 tabular datasets)."
    }
  ],
  cleaningSteps: [
    "Merged eight yearly files (195,054 rows).",
    "Removed 25,031 empty rows and 1 row with an invalid date, leaving 170,022 incidents.",
    "Parsed three different date formats, one per group of files.",
    "Harmonised column names and category spacing. Merged the two Ayutthaya labels (77 provinces).",
    "Set coordinates outside Thailand to missing.",
    "Kept casualty columns out of the features. Dropped 1,254 repeated accident codes (2023) for modelling.",
    "Found that the 2023 file spells many category values differently (about 82% of its presumed causes and 97% of its road locations never occur in 2019–2022). Kept as is and noted as a limitation."
  ],

  // Headline facts shown in the hero. Use audited values only.
  highlights: [
  {
    "value": "170,022",
    "label": "dated incidents after cleaning"
  },
  {
    "value": "2019–2026",
    "label": "eight yearly files, to 12 October 2026"
  },
  {
    "value": "77",
    "label": "provinces covered"
  },
  {
    "value": "11.4%",
    "label": "of incidents are fatal"
  }
],

  // state is "done", "current" or "upcoming".
  milestones: [
  {
    "id": "M1",
    "title": "Pitch the idea",
    "date": "14–16 October",
    "state": "done"
  },
  {
    "id": "M2",
    "title": "Lock the data",
    "date": "27 / 29 October",
    "state": "done"
  },
  {
    "id": "M3",
    "title": "Show progress",
    "date": "10 / 12 November",
    "state": "done"
  },
  {
    "id": "M4",
    "title": "Final submission",
    "date": "24 / 26 November",
    "state": "current"
  }
],

  // Edit team names and GitHub profiles here. Emails are not published (course rule: no personal data).
  team: [
  {
    "name": "Aung Ko Ko Minn",
    "github": "https://github.com/akkminn"
  },
  {
    "name": "Htoo Myat Htin Lynn",
    "github": "https://github.com/KingZenon"
  },
  {
    "name": "Thant Mg Mg Lwin",
    "github": "https://github.com/iYoon-Maiden"
  }
],

  data: {
  "status": "Locked on 29 October 2026 (audited 22 October)",
  "totalRecords": "170,022 dated incidents (195,054 rows in the files)",
  "dateRange": "1 January 2019 to 12 October 2026 (2026 is a partial year)",
  "targetDefinition": "Fatal 11.4% · Injury 45.0% · Property damage only 43.6%, derived from casualty counts",
  "cleaningSummary": "Merged eight yearly files. Removed 25,031 empty rows (21,052 in 2019, 3,979 in 2026) and one row with an invalid date. Parsed three different date formats per file, harmonised column names and category spacing, set coordinates outside Thailand to missing, and left out casualty columns from the features. For modelling, 1,254 repeated accident codes (2023) were dropped, leaving 168,768 incidents. The two labels used for Ayutthaya were merged, giving 77 provinces.",
  "years": [
    {
      "year": 2019,
      "records": "19,194",
      "dateRange": "2019-01-01 to 2019-12-31",
      "file": "accident2019.csv",
      "fileUrl": "https://datagov.mot.go.th/dataset/7e077ffd-dc4f-4dc6-a71c-0813726f3c12/resource/4d919e5b-aa06-4fe4-8a13-5571747a1254/download/accident2019.csv"
    },
    {
      "year": 2020,
      "records": "21,238",
      "dateRange": "2020-01-01 to 2020-12-31",
      "file": "accident2020.csv",
      "fileUrl": "https://datagov.mot.go.th/dataset/7e077ffd-dc4f-4dc6-a71c-0813726f3c12/resource/40476c7b-4194-4dc2-aba5-7326819ed071/download/accident2020.csv"
    },
    {
      "year": 2021,
      "records": "20,062",
      "dateRange": "2021-01-01 to 2021-12-31",
      "file": "accident2021.csv",
      "fileUrl": "https://datagov.mot.go.th/dataset/7e077ffd-dc4f-4dc6-a71c-0813726f3c12/resource/d0a8601a-3ee5-4ce8-8e9d-3dae3915dc30/download/accident2021.csv"
    },
    {
      "year": 2022,
      "records": "20,566",
      "dateRange": "2022-01-01 to 2022-12-31",
      "file": "accident2022.csv",
      "fileUrl": "https://datagov.mot.go.th/dataset/7e077ffd-dc4f-4dc6-a71c-0813726f3c12/resource/733b7874-bd5f-44b9-b271-650890b061f2/download/accident2022.csv"
    },
    {
      "year": 2023,
      "records": "23,841",
      "dateRange": "2023-01-01 to 2023-12-31",
      "file": "accident2023.csv",
      "fileUrl": "https://datagov.mot.go.th/dataset/7e077ffd-dc4f-4dc6-a71c-0813726f3c12/resource/661f2ead-1f28-4cd9-8a67-1bef76b33ef6/download/accident2023.csv"
    },
    {
      "year": 2024,
      "records": "23,952",
      "dateRange": "2024-01-01 to 2024-12-31",
      "file": "accident2024.csv",
      "fileUrl": "https://datagov.mot.go.th/dataset/7e077ffd-dc4f-4dc6-a71c-0813726f3c12/resource/83e9fae0-5f33-45e7-9a6d-6d5ad2060a08/download/accident2024.csv"
    },
    {
      "year": 2025,
      "records": "23,715",
      "dateRange": "2025-01-01 to 2025-12-31",
      "file": "accident2025.csv",
      "fileUrl": "https://datagov.mot.go.th/dataset/7e077ffd-dc4f-4dc6-a71c-0813726f3c12/resource/e1db5e93-2b70-4ee4-b6d7-7ead76d14a09/download/accident2025.csv"
    },
    {
      "year": 2026,
      "records": "17,454",
      "dateRange": "2026-01-01 to 2026-10-12 (partial year)",
      "file": "accident2026.csv",
      "fileUrl": "https://datagov.mot.go.th/dataset/7e077ffd-dc4f-4dc6-a71c-0813726f3c12/resource/4625b7aa-99a4-4dfb-9a0f-6402814d0f2c/download/accident2026.csv"
    }
  ],
  "columns": [
    {
      "original": "ปีที่เกิดเหตุ",
      "english": "Accident year",
      "note": "The accident date is used instead; year is not a model feature"
    },
    {
      "original": "วันที่เกิดเหตุ",
      "english": "Accident date",
      "note": "Three formats: day/month/year (2019, 2026), month/day/year (2020–2023), Excel serial number (2024–2025); parsed per file"
    },
    {
      "original": "เวลา",
      "english": "Accident time",
      "note": "Text such as 0:05, or a day fraction in 2024; converted to hour of day, with night flag"
    },
    {
      "original": "วันที่รายงาน / เวลาที่รายงาน",
      "english": "Report date and time",
      "note": "Not used as features"
    },
    {
      "original": "ACC_CODE",
      "english": "Accident code",
      "note": "2020–2025 only; 1,254 exact repeats in 2023 removed"
    },
    {
      "original": "หน่วยงาน",
      "english": "Responsible agency",
      "note": "Categorical feature"
    },
    {
      "original": "สายทาง",
      "english": "Route name",
      "note": "Not used: about 13% missing overall and 77% missing in 2024"
    },
    {
      "original": "KM / ก.ม.",
      "english": "Kilometre marker",
      "note": "Numeric feature"
    },
    {
      "original": "จังหวัด",
      "english": "Province",
      "note": "Categorical feature (about 1% missing)"
    },
    {
      "original": "รถคันที่ 1 / รถคันที่1",
      "english": "First vehicle type",
      "note": "Categorical feature; also defines the motorcycle subset"
    },
    {
      "original": "บริเวณที่เกิดเหตุ",
      "english": "Road location and geometry",
      "note": "Categorical feature (about 8% missing)"
    },
    {
      "original": "มูลเหตุสันนิษฐาน",
      "english": "Presumed cause",
      "note": "Categorical feature"
    },
    {
      "original": "ลักษณะการเกิดเหตุ / ลักษณะการเกิดอุบัติเหตุ",
      "english": "Accident type",
      "note": "Categorical feature (about 5% missing)"
    },
    {
      "original": "สภาพอากาศ",
      "english": "Weather",
      "note": "Categorical feature"
    },
    {
      "original": "LATITUDE / LONGITUDE",
      "english": "GPS coordinates",
      "note": "Values outside Thailand set to missing"
    },
    {
      "original": "รถที่เกิดเหตุ",
      "english": "Number of vehicles involved",
      "note": "Numeric feature"
    },
    {
      "original": "รถและคนที่เกิดเหตุ",
      "english": "Vehicles and persons involved",
      "note": "Numeric feature; absent in 2019"
    },
    {
      "original": "ผู้เสียชีวิต / จำนวนผู้เสียชีวิต",
      "english": "Deaths",
      "note": "Defines the fatal class; excluded from features"
    },
    {
      "original": "ผู้บาดเจ็บสาหัส / ผู้บาดเจ็บเล็กน้อย",
      "english": "Serious and minor injuries",
      "note": "Absent in 2019; excluded from features"
    },
    {
      "original": "รวมจำนวนผู้บาดเจ็บ",
      "english": "Total injured",
      "note": "Defines the injury class; excluded from features"
    }
  ]
},

  progress: [
    {
      "date": "24 November 2026",
      "title": "Follow-up improvements, literature audit & final slides",
      "done": "Added post hoc validation-only follow-up (2022–2025 folds; 2026 untouched): harmonised 2023 Thai spelling synonyms and added zero-leakage TRAMS features (road tier, holiday peaks, cyclical time, extra persons), lifting calibrated native LightGBM PR-AUC to 0.456 and fatal F1 to 0.466 (ECE 0.010; top-5% urgent tier 60.8% precision) and binary motorcycle ROC-AUC to 0.787 (exceeding 0.768 in all 4 folds). Verified open-access related-work details and linked the final presentation slides.",
      "issue": "Follow-up gains were validated post hoc on 2022–2025 folds only so the locked 2026 single-shot test remains leak-free. External rainfall, speed limits and traffic exposure are still outside TRAMS.",
      "next": "Final presentation delivery and public repository verification."
    },
    {
      "date": "24 November 2026",
      "title": "Re-check with stricter practice",
      "done": "An independent re-check on the validation years: native categorical handling, SMOTE-NC with standardised numbers, calibration, an inner-year cut-off, feature timing and error analysis. All four conclusions still hold. The 2026 test year was not touched.",
      "issue": "A plain model with a cut-off chosen on the last training year is as good as class weights, and class-weighted scores are not calibrated probabilities. The 2023 category spellings affect one fold.",
      "next": "Slides, final presentation, and uploading code, data and paper."
    },
    {
      "date": "23 November 2026",
      "title": "Milestone 4: final experiments and paper draft",
      "done": "Nested tuning and SMOTE-NC added; motorcycle subset and UK STATS19 replication run; the model chosen by a rule fixed beforehand and the 2026 test year scored once. Final paper drafted.",
      "issue": "SMOTE-NC and tuning added little, and the leading models are statistically comparable. On motorcycle crashes the best tuned model only matches the benchmark AUC.",
      "next": "Slides, final presentation, and uploading code, data and paper."
    },
    {
      "date": "19 November 2026",
      "title": "Error analysis and province map added",
      "done": "Confusion matrices, fatal precision–recall curves and grouped permutation importance for the 2025 validation fold, and a province map of fatal share. A check on the vehicles-and-persons-involved column found it does not drive the scores. All drawn from generated files.",
      "issue": "The plain and class-weighted precision–recall curves almost overlap: class weights move the cut-off, not the ranking.",
      "next": "Tuning, SMOTE, motorcycle subset, UK run, 2026 test score, paper draft."
    },
    {
      "date": "12 November 2026",
      "title": "Milestone 3: first results published",
      "done": "Cross-validation for 6 model families in 3 variants. Page redesigned. UK STATS19 (2021–2024) profiled.",
      "issue": "Higher fatal recall costs precision, and PR-AUC stays near 0.43.",
      "next": "Tuning, SMOTE, motorcycle subset, UK run, 2026 test score, paper draft."
    },
    {
      "date": "3 November 2026",
      "title": "Team name added",
      "done": "Team AHT on the page. Source files link to the publisher.",
      "next": "Merge the Ayutthaya labels, profile UK data, run cross-validation."
    },
    {
      "date": "29 October 2026",
      "title": "Dataset locked, data audit published",
      "done": "Eight files locked with SHA-256 hashes. Audit, severity target and related work added.",
      "next": "Cross-validation and imbalance experiments."
    },
    {
      "date": "26 October 2026",
      "title": "Website built and published",
      "done": "Page structure created and published on GitHub Pages.",
      "next": "Add the data audit and results."
    },
    {
      "date": "22 October 2026",
      "title": "First models trained (internal check)",
      "done": "Baselines trained on 2019–2024 and checked on 2025. Not published.",
      "next": "Add imbalance handling and tuning."
    },
    {
      "date": "22 October 2026",
      "title": "Dataset downloaded and audited",
      "done": "195,054 rows merged, three date formats handled, severity target derived.",
      "issue": "2019 lacks a serious/minor split. 2023 repeats 1,641 accident codes.",
      "next": "Run baseline models."
    },
    {
      "date": "22 October 2026",
      "title": "Benchmark paper read",
      "done": "Mohamad et al. (2025): motorcycle crashes only, random forest fatal recall 0.198.",
      "issue": "Different data, so not directly comparable.",
      "next": "Motorcycle-only comparison on our data."
    }
  ],

  // Scores, charts and the results table come from results-data.js (generated). Only text lives here.
  results: {
    "status": "Final results · 23 November 2026",
    "summary": "Rolling validation on 2022–2025, nested tuning, and one score on the 2026 test year. Class weights and prior adjustment raise fatal recall at a cost in precision; tuning and SMOTE-NC add little.",
    "takeaways": [
    "Every model beats the majority-class baseline, and boosted trees lead.",
    "Plain models find about a quarter of fatal incidents; class weights or prior adjustment more than double that, at a cost in precision.",
    "The plain and weighted precision–recall curves coincide: the strategies move the cut-off, not the ranking.",
    "SMOTE-NC and tuning added little, and the leading models are comparable (their differences are within the fold noise).",
    "A plain model with a cut-off chosen on the last training year performs about as well as class weights, and only the plain model's scores are calibrated probabilities.",
    "The same ordering of strategies holds on UK STATS19 data, where fatal crashes are much rarer."
  ],
    "limits": [
    "Only four rolling folds, and only XGBoost and LightGBM were tuned.",
    "The fatal share falls over time, and 2026 is a partial year (to 12 October).",
    "The data covers the Ministry of Transport road network only, and severity is derived from casualty counts.",
    "SMOTE-NC was tried in one configuration, on ordinal-coded categories; importance describes the model, not causes.",
    "The 2023 file spells many category values differently, which lowers that fold's recall and F1 but not its PR-AUC.",
    "Native categories, calibration and the cut-off rule were checked in a post hoc re-check on the validation years only.",
    "UK classes differ from the Thai ones, and the motorcycle comparison is not like for like."
  ],
    "draft": {
      "working": [
        "A reproducible pipeline from locked data to a single leak-free 2026 test score, replicated on UK STATS19.",
        "Confirmed negative result: SMOTE-NC (even scaled) and random tuning do not beat simpler calibrated or weighted boosted trees (supporting Elor & Averbuch-Elor, 2022).",
        "Solved calibration & alert precision trade-off on 2022–2025 folds: plain native LightGBM stays calibrated (ECE 0.010), reaches 60.8% precision on the top-5% urgent tier, and hits 0.466 fatal F1 with engineered TRAMS features.",
        "Dedicated binary motorcycle model with harmonised/engineered features reaches 0.787 ± 0.013 ROC-AUC (exceeding the 0.768 benchmark in all 4 rolling folds) at 0.460 fatal recall."
      ],
      "notWorking": [
        "At high recall (55%–66%), fatal precision stays at 35%–41% because TRAMS lacks crash-physics and occupant variables (impact speed, helmet/seatbelt use, rider age); use tiered alerts rather than automated penalties.",
        "Post hoc follow-up gains (0.466 fatal F1, 0.787 motorcycle AUC) were validated strictly on 2022–2025 folds to preserve the single-shot 2026 test lock.",
        "TRAMS covers MOT highways only and lacks external rainfall, road speed limits, and traffic exposure (vehicle-km)."
      ],
      "fixes": [
        "Deploy the plain calibrated model for risk scores with agency-selected alert tiers (e.g., top-5% urgent dispatch vs. broader review).",
        "Join external TMD/ERA5 rainfall, posted speed limits, and MOT/DLT vehicle-km exposure to break the remaining data ceiling.",
        "Deliver the final presentation and verify all public links before submission."
      ]
    },
    "figures": []
  },

  files: [
  {
    "label": "Project proposal (public copy)",
    "path": "assets/Project-Proposal-Public.pdf",
    "type": "PDF"
  },
  {
    "label": "Kickoff slides",
    "path": "assets/Predicting Road Accident Severity.pdf",
    "type": "PDF"
  },
  {
    "label": "ICT324 course brief",
    "path": "assets/ICT324-ProjectInfo.pdf",
    "type": "PDF"
  },
  {
    "label": "Dataset source",
    "path": "https://data.go.th/en/dataset/gdpublish-roadaccident",
    "type": "LINK"
  },
  {
    "label": "Verified dataset or data access instructions",
    "path": "https://github.com/KingZenon/research/blob/main/analysis/DATA.md",
    "type": "DATA"
  },
  {
    "label": "Dataset lock manifest (file sizes and SHA-256 hashes)",
    "path": "analysis/data-manifest.csv",
    "type": "DATA"
  },
  {
    "label": "Cross-validation results, summary (mean and standard deviation)",
    "path": "analysis/results/cv_temporal_thai_summary.csv",
    "type": "DATA"
  },
  {
    "label": "Cross-validation results, every fold",
    "path": "analysis/results/cv_temporal_thai_folds.csv",
    "type": "DATA"
  },
  {
    "label": "Confusion matrices, 2025 validation (every model and variant)",
    "path": "analysis/results/confusion_matrices_2025.csv",
    "type": "DATA"
  },
  {
    "label": "Fatal precision at each recall level, 2025 validation",
    "path": "analysis/results/pr_curves_fatal_2025.csv",
    "type": "DATA"
  },
  {
    "label": "Permutation importance, XGBoost with class weights",
    "path": "analysis/results/permutation_importance_2025.csv",
    "type": "DATA"
  },
  {
    "label": "Incidents and fatal share by province (with 95% range)",
    "path": "analysis/results/province_fatal_share.csv",
    "type": "DATA"
  },
  {
    "label": "Selection report: the leading candidates and their fold gaps",
    "path": "analysis/results/selection_report.csv",
    "type": "DATA"
  },
  {
    "label": "Cross-validation, every model and strategy (summary)",
    "path": "analysis/results/cv_thai_all_summary.csv",
    "type": "DATA"
  },
  {
    "label": "Tuning log (every configuration, every fold)",
    "path": "analysis/results/tuning_thai.csv",
    "type": "DATA"
  },
  {
    "label": "2026 test result, scored once (with bootstrap ranges)",
    "path": "analysis/results/test_2026_final.csv",
    "type": "DATA"
  },
  {
    "label": "UK STATS19 replication, summary",
    "path": "analysis/results/cv_uk_summary.csv",
    "type": "DATA"
  },
  {
    "label": "Motorcycle subset, summary",
    "path": "analysis/results/cv_motorcycle_summary.csv",
    "type": "DATA"
  },
  {
    "label": "Re-check: summary tables (native categories, calibration, cut-off, errors)",
    "path": "analysis/results_recheck/report_tables.md",
    "type": "DATA"
  },
  {
    "label": "Re-check: every run (experiment log)",
    "path": "analysis/results_recheck/experiment_log.csv",
    "type": "DATA"
  },
  {
    "label": "Re-check: follow-up features, alert tiers & binary motorcycle check",
    "path": "analysis/results_recheck/recheck_exp11_improvements.json",
    "type": "DATA"
  },
  {
    "label": "Analysis code and reproduction guide",
    "path": "https://github.com/KingZenon/research/tree/main/analysis",
    "type": "CODE"
  },
  {
    "label": "Final paper (IEEE PDF)",
    "path": "assets/deliverables/Team_AHT_ICT324_Final_Paper.pdf",
    "type": "PDF"
  },
  {
    "label": "Final presentation slides",
    "path": "assets/deliverables/Team_AHT_ICT324_Final_Slides.pdf",
    "type": "SLIDES"
  }
],

  references: [
  {
    "citation": "Ministry of Transport. Road accidents on the Ministry of Transport road network (TRAMS). Open Government Data of Thailand.",
    "url": "https://data.go.th/en/dataset/gdpublish-roadaccident"
  },
  {
    "citation": "Mohamad, I., JomnonKwao, S., & Ratanavaraha, V. (2025). Machine learning predictive performance in road accident severity: A case study from Thailand. Results in Engineering, 26, 104833.",
    "url": "https://doi.org/10.1016/j.rineng.2025.104833"
  },
  {
    "citation": "Champahom, T., Se, C., Watcharamaisakul, F., Jomnonkwao, S., Karoonsoontawong, A., & Ratanavaraha, V. (2024). Tree-based approaches to understanding factors influencing crash severity across roadway classes: A Thailand case study. IATSS Research, 48.",
    "url": "https://doi.org/10.1016/j.iatssr.2024.09.001"
  },
  {
    "citation": "Uttra, S., Champahom, T., Jomnonkwao, S., Se, C., Wisutwattanasak, P., & Ratanavaraha, V. (2026). Exploring risk factors for crash severity during Thailand's holiday travel: Machine learning exploration compared to heterogeneity modeling. Future Transportation, 6(1), 39.",
    "url": "https://doi.org/10.3390/futuretransp6010039"
  },
  {
    "citation": "Phojaem, T., Se, C., Wisutwattanasak, P., Champahom, T., Jomnonkwao, S., & Ratanavaraha, V. (2026). Comparative injury-severity modeling on highway-speed motorways: Performance, interpretability, and policy implications. Case Studies on Transport Policy, 24.",
    "url": "https://doi.org/10.1016/j.cstp.2026.101826"
  },
  {
    "citation": "Simmachan, T., & Boonkrong, P. (2025). Effect of resampling techniques on machine learning models for classifying road accident severity in Thailand. Journal of Current Science and Technology, 15(2).",
    "url": "https://ph04.tci-thaijo.org/index.php/JCST/article/view/6443"
  },
  {
    "citation": "Al Mamun, A., Enan, A., Indah, D. A., Mwakalonge, J., Comert, G., & Chowdhury, M. (2025). Crash severity risk modeling strategies under data imbalance. arXiv preprint arXiv:2412.02094.",
    "url": "https://arxiv.org/abs/2412.02094"
  },
  {
    "citation": "Zhu, D., Xie, C., Wang, Z., & Zhang, H. (2025). RaX-Crash: A resource efficient and explainable small model pipeline with an application to city scale injury severity prediction. arXiv preprint arXiv:2512.07848.",
    "url": "https://arxiv.org/abs/2512.07848"
  },
  {
    "citation": "Elor, Y., & Averbuch-Elor, H. (2022). To SMOTE, or not to SMOTE? arXiv preprint arXiv:2201.08528.",
    "url": "https://arxiv.org/abs/2201.08528"
  },
  {
    "citation": "Sufian, M. A., & Varadarajan, J. (2023). Enhancing prediction and analysis of UK road traffic accident severity using AI: Integration of machine learning, econometric techniques, and time series forecasting in public health research. arXiv preprint arXiv:2309.13483.",
    "url": "https://arxiv.org/abs/2309.13483"
  },
  {
    "citation": "Tong, Y. (2025). Non-linear determinants of pedestrian injury severity: Evidence from administrative data in Great Britain. arXiv preprint arXiv:2512.04022.",
    "url": "https://arxiv.org/abs/2512.04022"
  },
  {
    "citation": "Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. Patterns, 4(9), 100804.",
    "url": "https://doi.org/10.1016/j.patter.2023.100804"
  },
  {
    "citation": "Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? Advances in Neural Information Processing Systems 35 (Datasets and Benchmarks Track).",
    "url": "https://arxiv.org/abs/2207.08815"
  },
  {
    "citation": "Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16, 321–357.",
    "url": "https://doi.org/10.1613/jair.953"
  },
  {
    "citation": "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794.",
    "url": "https://doi.org/10.1145/2939672.2939785"
  },
  {
    "citation": "Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems 30, 3146–3154."
  },
  {
    "citation": "Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLoS ONE, 10(3), e0118432.",
    "url": "https://doi.org/10.1371/journal.pone.0118432"
  },
  {
    "citation": "Wirth, R., & Hipp, J. (2000). CRISP-DM: Towards a standard process model for data mining. Proceedings of the 4th International Conference on the Practical Applications of Knowledge Discovery and Data Mining, 29–39."
  },
  {
    "citation": "Department for Transport (UK). Road safety data (STATS19), collision files 2021–2024.",
    "url": "https://www.gov.uk/government/statistics/road-safety-data"
  },
  {
    "citation": "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.",
    "url": "https://jmlr.org/papers/v12/pedregosa11a.html"
  },
  {
    "citation": "Lemaître, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn: A Python toolbox to tackle the curse of imbalanced datasets in machine learning. Journal of Machine Learning Research, 18(17), 1–5.",
    "url": "https://jmlr.org/papers/v18/16-365.html"
  }
]
};
