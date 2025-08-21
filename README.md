# Kaggle Playground Series S5E8 — Bank Term Deposit Prediction

This repository contains solutions for the Kaggle competition  
[Playground Series — Season 5, Episode 8](https://www.kaggle.com/competitions/playground-series-s5e8/leaderboard#).  
The goal is to predict whether a client will subscribe to a term deposit.

---

## 📌 Project Structure

```
.
├── notebooks/                # Jupyter notebooks (EDA, models, feature engineering, stacking)
├── src/                      # Source code
│   ├── config/               # YAML configs for CV, data, models
│   ├── features/             # Feature engineering scripts
│   ├── models/               # Training pipeline
│   ├── utils/                # IO and helper functions
│   ├── visualization/        # Plotting utilities
│   └── evaluation/           # Evaluation utilities
├── experiments/              # Saved configs, metrics, and predictions for experiments
├── submissions/              # Example submissions
├── docs/FEATURES.md          # Documentation of features
├── environment.yml           # Conda environment
├── Makefile                  # Automation commands
└── README.md                 # This file
```

---

## 📊 Data

The dataset is **not included** in this repository.  
You need to download it directly from Kaggle:  
👉 [Competition page](https://www.kaggle.com/competitions/playground-series-s5e8/data)  
👉 [Bank Marketing Dataset](https://www.kaggle.com/datasets/sushant097/bank-marketing-dataset-full?select=bank-full.csv)

After downloading, place the files into:

```
data/raw/               # Kaggle competition dataset
data/processed/         # processed datasets
data/original_dataset/  # original [Bank Marketing Dataset](https://www.kaggle.com/datasets/sushant097/bank-marketing-dataset-full?select=bank-full.csv)
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/kaggle-bank-term-deposit-2025.git
cd kaggle-bank-term-deposit-2025
```

### 2. Create environment
Using Conda:
```bash
conda env create -f environment.yml
conda activate bank-auc
```

### 3. Download data
Place Kaggle data into `data/raw/` as described above.

### 4. Run notebooks
You can explore experiments in the `notebooks/` folder:
- `00_eda.ipynb` → initial data exploration
- `01_baseline.ipynb` → simple baseline model
- `02_feature_engineering.ipynb` → feature engineering
- `03_catboost.ipynb`, `04_lgbm.ipynb`, `05_xgboost.ipynb` → model training
- `09_stacking_data.ipynb`, `10_meta_model_stacking.ipynb` → model stacking

---

## ⚙️ Experiments

All experiments are logged in the `experiments/` folder, including:
- model parameters (`*.yml`)
- validation predictions (`val_pred.parquet`)
- evaluation metrics (`metrics.yml`)

---

## 🔬 Approach

We followed an iterative process to improve the solution:

1. **Baseline**: trained a simple CatBoost model to establish a starting point.  
2. **Feature Engineering**: created additional features, including clustering-based ones, to enrich the dataset.  
3. **Hyperparameter Tuning**: optimized CatBoost, LightGBM, and XGBoost models with grid/random search.  
4. **Stacking**: combined multiple models into an ensemble to further improve the performance.  

Detailed metrics for each experiment can be found in the `experiments/` folder.

---

## 📈 Results

Best results are achieved with model stacking (CatBoost + LGBM + XGBoost).  
Final submissions can be found in `submissions/`.

---

## 📌 Notes

- The repository does **not** contain competition data (per Kaggle rules).  
- You must download it manually.  
- The repo is structured to support reproducibility: configs + scripts + experiments.

---

## 👤 Author

Developed by Daria Morgalenko.  
Feel free to open issues or contribute.
