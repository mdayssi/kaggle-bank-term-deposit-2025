# Kaggle Playground 2025 — Bank Term Deposit (ROC AUC)

Portfolio-ready, **clean** and **staged** project template. We start from an empty skeleton and progressively add code (with code review).

## Staged roadmap

**Milestone 0 — Scaffold (this commit)**
- Repo hygiene: structure, env, pre-commit, configs placeholders.

**Milestone 1 — Data & EDA**
- Add data schema checks (`src/data/schema.py`).
- Create `notebooks/00_eda.ipynb` (clean EDA only).
- No models here.

**Milestone 2 — Features**
- Extract stable feature logic into `src/features/`.
- Demo in `notebooks/01_feature_engineering.ipynb`.

**Milestone 3 — Baselines**
- LightGBM / XGBoost / CatBoost (`src/models/train.py`), unified CV, OOF saving.
- `notebooks/02_baselines.ipynb` only calls entrypoints and visualizes results.

**Milestone 4 — Tuning & Blending**
- Random/Optuna search; move best params into YAML under `src/config/`.
- OOF-based blending; final submission script.

**Milestone 5 — Interpretability & README polish**
- SHAP on sample, feature importance, conclusions section in README.

## How to use this repo

1) Create env (conda):
```
conda env create -f environment.yml
conda activate bank-auc
pre-commit install
```

2) Put Kaggle CSVs into `data/raw/` (ignored by git).

3) Follow milestones; each milestone should be a separate PR/commit.
