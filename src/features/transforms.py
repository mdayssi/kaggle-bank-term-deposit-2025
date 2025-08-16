from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, RobustScaler

import src.features.clustering as ctr
import src.features.basic as ftr_basic

MONTH2NUM = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

K_CANDIDATES = [4, 6, 8]
GAMMA_CANDIDATES = [0.5, 1.0, 2.0]
N_INIT = 3
MAX_ITER = 50
SEED = 42
N_SAMPLE_COST = 40000
N_SAMPLE_SIL = 8000


class JobBalanceEnc(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=10):
        self.smoothing_ = smoothing
        self.global_mean_ = None
        self.job_stats_ = None

    def fit(self, X: pd.DataFrame):
        df = X.copy()
        df["job"] = df["job"].astype(str)
        self.global_mean_ = df['balance'].mean()
        agg = df.groupby(by='job')['balance'].agg(['mean', 'count'])
        self.job_stats_ = (agg['mean'] * agg['count'] + self.global_mean_ * self.smoothing_) / (
                agg['count'] + self.smoothing_)
        return self

    def transform(self, X: pd.DataFrame):
        df = X.copy()
        df["job"] = df["job"].astype(str)
        df['job_balance_mean'] = df['job'].map(self.job_stats_)
        df['job_balance_mean'].fillna(self.global_mean_, inplace=True)
        return df[['job_balance_mean']]


def was_contact(df: pd.DataFrame) -> pd.Series:
    wc = (df['pdays'] != -1).astype(int)
    return wc


def credit_score(df: pd.DataFrame) -> pd.Series:
    mapping = {'yes': 1, 'no': 0}
    cs = df['loan'].astype(str).map(mapping) + \
         df['housing'].astype(str).map(mapping) + \
         df['default'].astype(str).map(mapping)
    return cs


def job_marital(df: pd.DataFrame) -> pd.Series:
    jm = df['job'].astype(str) + '_' + df['marital'].astype(str)
    return jm


def job_education(df: pd.DataFrame) -> pd.Series:
    je = df['job'].astype(str) + '_' + df['education'].astype(str)
    return je


def education_marital(df: pd.DataFrame) -> pd.Series:
    em = df['education'].astype(str) + '_' + df['marital'].astype(str)
    return em


def binning_categorical(column: pd.Series, bins: List[int | float], labels: List[str]) -> pd.Series:
    bins = pd.cut(
        column,
        bins=bins,
        labels=labels,
    )
    return bins


def pdays_categorical(df: pd.DataFrame) -> pd.Series:
    pdcat = binning_categorical(
        df['pdays'],
        bins=[-float("inf"), -1, 100, 300, float("inf")],
        labels=["no_contact", "<100", "100-300", "300+"],
    )
    return pdcat


def campaign_categorical(df: pd.DataFrame) -> pd.Series:
    campcat = binning_categorical(
        df["campaign"],
        bins=[0, 1, 2, 3, 5, 10, float("inf")],
        labels=["1", "2", "3", "3-5", "5-10", "10+"],
    )
    return campcat


def previous_categorical(df: pd.DataFrame) -> pd.Series:
    prcat = binning_categorical(
        df['previous'],
        bins=[-float("inf"), 0, 5, float("inf")],
        labels=["0", "<5", "5+"],
    )
    return prcat


def log_duration(df: pd.DataFrame) -> pd.Series:
    ld = np.log1p(df["duration"])
    return ld


def log_balance(df: pd.DataFrame) -> pd.Series:
    lb = np.log1p(df["balance"].clip(lower=0))
    return lb


def multiply_logs(df: pd.DataFrame) -> pd.Series:
    logd = log_duration(df)
    logb = log_balance(df)
    return logd * logb


def is_overdraft(df: pd.DataFrame) -> pd.Series:
    draft = (df["balance"] < 0).astype(int)
    return draft


def sin_month(df: pd.DataFrame) -> pd.Series:
    month_num = df["month"].astype(str).map(MONTH2NUM)
    sinm = np.sin(2 * np.pi * (month_num - 1) / 12)
    return sinm


def cos_month(df: pd.DataFrame) -> pd.Series:
    month_num = df["month"].astype(str).map(MONTH2NUM)
    cosm = np.cos(2 * np.pi * (month_num - 1) / 12)
    return cosm


def sin_day(df: pd.DataFrame) -> pd.Series:
    sind = np.sin(2 * np.pi * (df["day"] - 1) / 31)
    return sind


def cos_day(df: pd.DataFrame) -> pd.Series:
    cosd = np.cos(2 * np.pi * (df["day"] - 1) / 31)
    return cosd


def cluster_feature(df_train: pd.DataFrame, df_val: pd.DataFrame, num_robust, num_minmax) -> Tuple[
    pd.Series, pd.Series]:
    column_transformer = ColumnTransformer(
        [("robust", RobustScaler(), num_robust), ("minmax", MinMaxScaler(), num_minmax)],
        remainder="passthrough",
    )

    column_transformer.set_output(transform="pandas")
    trans_num_data_train = df_train.copy()
    trans_num_data_val = df_val.copy()
    trans_num_data_train = column_transformer.fit_transform(trans_num_data_train)
    trans_num_data_val = column_transformer.transform(trans_num_data_val)

    names_features = ftr_basic.get_features_names(trans_num_data_train)
    trans_cat_features = names_features["categorical"]
    trans_num_features = names_features["numeric"]

    trans_num_data_train = ftr_basic.cat_features_to_category(trans_num_data_train)
    trans_num_data_val = ftr_basic.cat_features_to_category(trans_num_data_val)

    trans_num_data_train, trans_num_data_val = ftr_basic.align_categorical_levels(
        trans_num_data_train, trans_num_data_val, trans_cat_features
    )

    print(">> Tuning k-prototypes (quick)...")
    tuning_table = ctr.tune_kproto_fast(
        trans_num_data_train, trans_num_features, trans_cat_features,
        K_list=K_CANDIDATES, gamma_list=GAMMA_CANDIDATES,
        n_sample_cost=N_SAMPLE_COST, n_sample_sil=N_SAMPLE_SIL,
        n_init=N_INIT, max_iter=MAX_ITER, seed=SEED
    )
    print(tuning_table.head(10))
    bestK = tuning_table.iloc[0]['K']
    bestG = tuning_table.iloc[0]['gamma']
    print(f">> Best (quick): K={bestK}, gamma={bestG}, "
          f"sil={tuning_table.iloc[0]['silhouette']:.4f}, cost={tuning_table.iloc[0]['cost']:.0f}")

    print(">> Fitting final k-prototypes on full train...")
    kp, labels_tr, labels_va = ctr.fit_final_kproto(
        trans_num_data_train, trans_num_data_val,
        trans_num_features, trans_cat_features,
        K=int(bestK), gamma=float(bestG), seed=SEED
    )

    cluster_tr = pd.Series(labels_tr, index=trans_num_data_train.index, name='cluster_kp').astype('category')
    cluster_va = pd.Series(labels_va, index=trans_num_data_val.index, name='cluster_kp').astype('category')

    return cluster_tr, cluster_va
