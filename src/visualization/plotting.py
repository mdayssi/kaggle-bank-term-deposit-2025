import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from typing import List, Any

plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": 100,
    }
)

sns.set_theme(style="whitegrid", palette="muted")


def plotting_histogram_numerical_features(data: pd.DataFrame, num_features: List[str], color="#69b3a2"):
    data = data[num_features]
    cnt_ftr = len(num_features)
    ncols = 2
    nrows = (cnt_ftr + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 10))
    axs = axs.flatten()
    for i, feature in enumerate(num_features):
        bins = range(int(data[feature].min()), int(data[feature].max()) + 1) if len(data[feature].unique()) < 50 else 50
        sns.histplot(data=data[feature], bins=bins, ax=axs[i], kde=True, color=color)
        axs[i].set_title(f"{feature}")
        axs[i].set_xlabel("")
    for ax in axs[cnt_ftr:]:
        ax.remove()
    plt.suptitle(f"Histograms of numerical features")
    plt.tight_layout()
    plt.show()


def plotting_boxplot_numerical_features(data: pd.DataFrame, num_features: List[str], color="#69b3a2"):
    data = data[num_features]
    cnt_ftr = len(num_features)
    ncols = 4
    nrows = (cnt_ftr + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    axs = axs.flatten()
    for i, feature in enumerate(num_features):
        sns.boxplot(y=data[feature], ax=axs[i], color=color)
        axs[i].set_title(f"{feature}")
        axs[i].set_xlabel("")
    for ax in axs[cnt_ftr:]:
        ax.remove()
    plt.suptitle(f"Box-plot of numerical features")
    plt.tight_layout()
    plt.show()


def plotting_countplot_categorical_features(data: pd.DataFrame, cat_features: List[str], color="#F4A460"):
    data = data[cat_features]
    cnt_ftr = len(cat_features)
    rotate_threshold = 6
    ncols = 3
    nrows = (cnt_ftr + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 12))
    axs = axs.flatten()
    for i, feature in enumerate(cat_features):
        sns.countplot(x=data[feature], ax=axs[i], color=color)
        axs[i].set_title(f"{feature}")
        axs[i].set_xlabel("")
        n_labels = len(axs[i].get_xticklabels())
        # angle = 45 if n_labels > rotate_threshold else 0
        # axs[i].tick_params(axis='x', rotation=angle, labelsize=9)
        if n_labels > rotate_threshold:
            for lbl in axs[i].get_xticklabels():
                lbl.set_rotation(45)
                lbl.set_horizontalalignment('right')  # ha='right'
                lbl.set_rotation_mode('anchor')
    for ax in axs[cnt_ftr:]:
        ax.remove()
    plt.suptitle(f"Histograms of categorical features")
    plt.tight_layout()
    plt.show()


def plotting_heatmap_cat_features(data: pd.DataFrame, target: pd.Series, cat_features: List[str]):
    data = data[cat_features]
    cnt_ftr = len(cat_features)
    # rotate_threshold = 6
    ncols = 2
    nrows = (cnt_ftr + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 20))
    axs = axs.flatten()
    for i, feature in enumerate(cat_features):
        crosstab = pd.crosstab(data[feature], target, normalize='index')
        sns.heatmap(crosstab, annot=True, fmt='.2f', ax=axs[i])
        axs[i].set_title(f"{feature}")
    for ax in axs[cnt_ftr:]:
        ax.remove()
    plt.suptitle(f"Heatmap of categorical features")
    plt.tight_layout()
    plt.show()


def plotting_kde_per_target(data: pd.DataFrame, target: pd.Series, num_features: List[str]):
    data = data[num_features]
    cnt_ftr = len(num_features)
    ncols = 2
    nrows = (cnt_ftr + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 20))
    axs = axs.flatten()
    for i, feature in enumerate(num_features):
        sns.kdeplot(data=data[target == 0], x=feature, label='y=0', fill=True, alpha=0.4, ax=axs[i])
        sns.kdeplot(data=data[target == 1], x=feature, label='y=1', fill=True, alpha=0.4, ax=axs[i])
        axs[i].set_title(f"{feature}")
        axs[i].legend()
        axs[i].set_xlabel("")
    for ax in axs[cnt_ftr:]:
        ax.remove()
    plt.suptitle(f"KDE numeric features per classes")
    plt.tight_layout()
    plt.show()


def shap_values(model: Any, df: pd.DataFrame, is_xgb=False) -> pd.DataFrame:
    if is_xgb:
        df_dmatrix = xgb.DMatrix(df, enable_categorical=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_dmatrix)
    shap.summary_plot(shap_values, df, plot_type="bar")

    mean_shap_values = np.abs(shap_values).mean(axis=0)

    df_imp = pd.DataFrame(
        {
            "feature": df.columns,
            "shap_value": mean_shap_values,
        }
    ).sort_values("shap_value", ascending=False)

    return df_imp
