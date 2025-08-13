from sklearn.base import BaseEstimator, TransformerMixin
from pandas import DataFrame
import pandas as pd
from typing import List



class JobBalanceEnc(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=10):
        self.smoothing_ = smoothing
        self.global_mean_ = None
        self.job_stats_ = None

    def fit(self, X: DataFrame):
        df = X.copy()
        self.global_mean_ = df['balance'].mean()
        agg = df.groupby(by='job')['balance'].agg(['mean', 'count'])
        self.job_stats_ = (agg['mean'] * agg['count'] + self.global_mean_ * self.smoothing_) / (
                agg['count'] + self.smoothing_)
        return self

    def transform(self, X: DataFrame):
        df = X.copy()
        df['job_balance_mean'] = df['job'].map(self.job_stats_)
        df['job_balance_mean'].fillna(self.global_mean_, inplace=True)
        return df[['job_balance_mean']]


def was_contact(df):
    wc = (df['pdays'] != -1).astype(int)
    return wc


def credit_score(df):
    mapping = {'yes': 1, 'no': 0}
    cs = df['loan'].map(mapping) + df['housing'].map(mapping) + df['default'].map(mapping)
    return cs


def job_marital(df):
    jm = df['job'].astype(str) + '_' + df['marital'].astype(str)
    return jm


def binning_categorical(column: pd.Series, bins: List[int | float], labels: List[str]):
    bins = pd.cut(
        column,
        bins=bins,
        labels=labels,
    )
    return bins


def pdays_categorical(df: pd.DataFrame):
    pdcat = binning_categorical(
        df['pdays'],
        bins=[-float("inf"), -1, 100, 300, float("inf")],
        labels=["no_contact", "<100", "100-300", "300+"],
    )
    return pdcat


def campaign_categorical(df: pd.DataFrame):
    campcat = binning_categorical(
        df["campaign"],
        bins=[0, 1, 2, 3, 5, 10, float("inf")],
        labels=["1", "2", "3", "3-5", "5-10", "10+"],
    )
    return campcat


def previous_categorical(df: pd.DataFrame):
    prcat = binning_categorical(
        df['previous'],
        bins=[-float("inf"), 0, 5, float("inf")],
        labels=["0", "<5", "5+"],
    )
    return prcat