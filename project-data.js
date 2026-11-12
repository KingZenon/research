/*
 * Edit this file when the team supplies data, results, and final materials.
 * Keep unknown values as null. Do not enter estimated record counts or scores.
 * See README.md for the exact update steps and file locations.
 */
window.PROJECT_DATA = {
  lastUpdated: "12 November 2026",
  teamName: "Team AHT",

  // Short related-work items and cleaning steps (one line each).
  relatedWork: [
    {
      "cite": "Mohamad et al. (2025)",
      "text": "Thai motorcycle crashes, 2015–2020. Random forest was best but caught only 20% of fatal crashes."
    },
    {
      "cite": "Champahom et al. (2024)",
      "text": "Thai roadway classes. XGBoost was strongest on highways."
    },
    {
      "cite": "Uttra et al. (2026)",
      "text": "Thai holiday crashes in three severity classes. XGBoost was best (83.8% accuracy)."
    },
    {
      "cite": "Phojaem et al. (2026)",
      "text": "Injury severity on highway-speed motorways, several models compared."
    }
  ],
  cleaningSteps: [
    "Merged eight yearly files (195,054 rows).",
    "Removed 25,031 empty rows and 1 row with an invalid date, leaving 170,022 incidents.",
    "Parsed three different date formats, one per group of files.",
    "Harmonised column names and category spacing. Merged the two Ayutthaya labels (77 provinces).",
    "Set coordinates outside Thailand to missing.",
    "Kept casualty columns out of the features. Dropped 1,254 repeated accident codes (2023) for modelling."
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
    "state": "current"
  },
  {
    "id": "M4",
    "title": "Final submission",
    "date": "24 / 26 November",
    "state": "upcoming"
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
    "status": "First results · 12 November 2026",
    "summary": "Four rolling folds, validated on 2022–2025. Class weights raise fatal recall at a cost in precision.",
    "takeaways": [
    "Every model beats the majority-class baseline (macro-F1 0.48–0.54 against 0.205).",
    "Plain models find only 14–26% of fatal incidents.",
    "Class weights or prior adjustment lift that to 0.54–0.68 for most models, but fatal precision falls to about 0.26–0.37.",
    "PR-AUC does not change (about 0.43 for the best models): the cut-off moves, the ranking does not improve."
  ],
    "limits": [
    "Untuned settings and only four folds.",
    "The fatal share falls over time (13.8% to 9.9%), and the data has no rider or driver variables.",
    "The 2026 test year is not scored yet."
  ],
    "draft": {
      "working": [
        "Reproducible pipeline: eight locked files and 170,022 dated incidents.",
        "Every model beats the majority-class baseline.",
        "Class weights more than double fatal recall for XGBoost and LightGBM in every validation year."
      ],
      "notWorking": [
        "Fatal precision drops to about 0.35 when recall rises.",
        "PR-AUC stays near 0.43 for the best models: ranking is not improving.",
        "Injury and property damage only are often confused.",
        "No tuning, SMOTE, test-year score or UK run yet."
      ],
      "fixes": [
        "Tune inside the training years and add SMOTE.",
        "Engineer new features (holidays, road class, location history).",
        "Run a motorcycle-only subset and the UK STATS19 data.",
        "Pick one model on validation, score 2026 once, explain with SHAP."
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
    "label": "Analysis code and reproduction guide",
    "path": "https://github.com/KingZenon/research/tree/main/analysis",
    "type": "CODE"
  },
  {
    "label": "Final paper",
    "path": null,
    "type": "PDF"
  },
  {
    "label": "Final presentation",
    "path": null,
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
  }
]
    // Add at least 10 checked research references with complete citation and URL.
};
