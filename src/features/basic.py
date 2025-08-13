from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


def get_features_names(df: pd.DataFrame) -> Dict[str, List[str]]:
    cat_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_features = df.select_dtypes(include=[np.number]).columns.tolist()
    return {'categorical': cat_features, 'numeric': num_features}


def cat_features_to_category(df: pd.DataFrame) -> pd.DataFrame:
    out_df = df.copy()
    object_columns = out_df.select_dtypes(include=['object']).columns.tolist()
    out_df.loc[:, object_columns] = out_df.loc[:, object_columns].astype('category')
    return out_df

def align_categorical_levels(df_train: pd.DataFrame, df_val: pd.DataFrame, cat_features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df_train.copy()
    val = df_val.copy()
    for col in cat_features:
        levels = train[col].cat.categories.union(val[col].cat.categories)
        train[col] = train[col].cat.set_categories(levels)
        val[col] = val[col].cat.set_categories(levels)
    return train, val






