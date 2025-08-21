import numpy as np
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from typing import Dict, Any, Tuple
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from tqdm.auto import tqdm

RANDOM_SEED = 42


def sample_params(n_comb=24, random_seed=42):
    rng = np.random.default_rng(random_seed)
    params = []
    for _ in range(n_comb):
        param = {
            'learning_rate': float(rng.choice([0.02, 0.03, 0.05])),
            'num_leaves': int(rng.integers(31, 128)),
            'min_data_in_leaf': int(rng.integers(20, 200)),
            'reg_alpha': float(10 ** rng.uniform(-2, 0)),
            'reg_lambda': float(10 ** rng.uniform(-2, 1)),
            'feature_fraction': float(rng.uniform(0.7, 1.0)),
            'bagging_fraction': float(rng.uniform(0.7, 1.0)),
            'bagging_freq': int(rng.integers(1, 5)),
            'max_cat_to_onehot': 6,
            'cat_l2': float(10 ** rng.uniform(-1, 2)),
            'cat_smooth': float(10 ** rng.uniform(0, 3)),
        }
        params.append(param)
    return params


def cv_lgbm(X, y, cat_features, params, n_splits=3, random_seed=RANDOM_SEED):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    aucs = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)], eval_metric='auc',
            categorical_feature=cat_features,
            callbacks=[early_stopping(200, verbose=False), log_evaluation(0)]
        )

        pred = model.predict_proba(X_val, raw_score=False)[:, 1]
        aucs.append(roc_auc_score(y_val, pred))
    print(f'mean auc for params: {float(np.mean(aucs))}')
    return float(np.mean(aucs)), float(np.std(aucs))


def random_search_cv_lgbm(train_pool, cat_features, base_params, val_pool=None, n_iter=24,
                          cv=3, get_params=False, refit=False, random_seed=RANDOM_SEED):
    X_train, y_train = train_pool
    X_train = X_train.copy()
    if val_pool is not None:
        X_val, y_val = val_pool
        X_val = X_val.copy()

    candidates = sample_params(n_iter, random_seed)
    max_auc = 0
    best_params = {}
    if get_params:
        results = []
    for i, param in enumerate(candidates):
        print(f'=============== comb params {i + 1}/{n_iter} ===============')
        all_params = {**base_params, **param}
        mean_auc, std_auc = cv_lgbm(X_train, y_train, cat_features, all_params, n_splits=cv,
                                    random_seed=RANDOM_SEED)
        if get_params:
            results.append({**param, 'cv_auc': mean_auc, 'cv_std': std_auc})
        if mean_auc > max_auc:
            max_auc = mean_auc
            best_params = param
    if refit:
        if val_pool is None:
            raise ValueError("refit=True требует val_pool=(X_val, y_val) для early stopping.")
        best_model = LGBMClassifier(
            **base_params,
            **best_params,
        )

        best_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)], eval_metric='auc',
            categorical_feature=cat_features,
            callbacks=[early_stopping(200, verbose=True)]
        )
        return best_model
    if get_params:
        results = pd.DataFrame(results).sort_values('cv_auc', ascending=False)
        return results
    return best_params


# --- XGBoost randomized CV ---

class XGBBoosterWrapper:
    def __init__(self, booster: xgb.Booster, params: dict):
        self.booster = booster
        self.params_ = dict(params)
        # best_iteration is set by xgb.train when early stopping is used
        self.best_iteration = getattr(booster, "best_iteration", None)

    def get_params(self, deep=False):
        # для печати лучших гиперов
        return dict(self.params_)

    def predict_proba(self, X):
        d = xgb.DMatrix(X, enable_categorical=True)
        # в 3.0.3 поддерживается iteration_range; оставим fallback на ntree_limit
        if self.best_iteration is not None:
            try:
                pred = self.booster.predict(d, iteration_range=(0, self.best_iteration + 1))
            except TypeError:
                pred = self.booster.predict(d, ntree_limit=self.best_iteration + 1)
        else:
            pred = self.booster.predict(d)
        # вернуть 2 столбца (как у sklearn): [1-p, p]
        return np.column_stack([1.0 - pred, pred])


def sample_params_xgb(n_comb=24, random_seed=42):
    rng = np.random.default_rng(random_seed)
    params = []
    for _ in range(n_comb):
        param = {
            "learning_rate": float(10 ** rng.uniform(-2.3, -0.7)),  # ~[0.005, 0.2]
            "max_depth": int(rng.integers(3, 9)),  # 3..8
            "min_child_weight": float(10 ** rng.uniform(0, 1)),  # [1, 10]
            "subsample": float(rng.uniform(0.6, 1.0)),
            "colsample_bytree": float(rng.uniform(0.6, 1.0)),
            "reg_alpha": float(10 ** rng.uniform(-3, 0)),  # [1e-3, 1]
            "reg_lambda": float(10 ** rng.uniform(-3, 1)),  # [1e-3, 10]
            "gamma": float(10 ** rng.uniform(-3, 0)),  # [1e-3, 1]
            "max_cat_to_onehot": int(rng.integers(4, 16)),  # для enable_categorical
        }
        params.append(param)
    return params


def cv_xgb(X, y, cat_features, params, n_splits=3, random_seed=RANDOM_SEED):
    # гарантируем нужные dtypes для категорий
    X = X.copy()
    for col in cat_features:
        X[col] = X[col].astype("category")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    aucs = []

    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        pos = y_tr.sum()
        neg = len(y_tr) - pos
        scale_pos_weight = float(neg / pos)

        # базовые и sampled-параметры
        xgb_params = {
            "scale_pos_weight": scale_pos_weight,
            **params,
        }

        dtr = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
        dva = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

        booster = xgb.train(
            params=xgb_params,
            dtrain=dtr,
            num_boost_round=10000,
            evals=[(dva, "val")],
            early_stopping_rounds=200,
            verbose_eval=False,
        )

        # предикт на лучшей итерации
        try:
            p = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
        except TypeError:
            p = booster.predict(dva, ntree_limit=booster.best_iteration + 1)

        aucs.append(roc_auc_score(y_val, p))

    print(f"mean auc for params: {float(np.mean(aucs))}")
    return float(np.mean(aucs)), float(np.std(aucs))


def random_search_cv_xgb(train_pool, cat_features, base_params, val_pool=None, num_comb=24, n_splits=3,
                         get_params=False, refit=False, random_seed=RANDOM_SEED):
    X_train, y_train = train_pool
    X_train = X_train.copy()
    if val_pool is not None:
        X_val, y_val = val_pool
        X_val = X_val.copy()

    candidates = sample_params_xgb(num_comb, random_seed)
    max_auc = 0.0
    best_params = {}
    if get_params:
        results = []

    for i, param in enumerate(candidates, start=1):
        print(f"=============== comb params {i}/{num_comb} ===============")
        param = {**base_params, **param}
        mean_auc, std_auc = cv_xgb(X_train, y_train, cat_features, param,
                                   n_splits=n_splits, random_seed=random_seed)
        if get_params:
            results.append({**param, "cv_auc": mean_auc, "cv_std": std_auc})
        if mean_auc > max_auc:
            max_auc = mean_auc
            best_params = param

    if refit:
        if val_pool is None:
            raise ValueError("refit=True требует val_pool=(X_val, y_val) для early stopping.")

        pos = y_train.sum()
        neg = len(y_train) - pos
        scale_pos_weight = float(neg / pos)

        xgb_params = {
            **base_params,
            "scale_pos_weight": scale_pos_weight,
            **best_params,
        }

        dtr = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dva = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

        booster = xgb.train(
            params=xgb_params,
            dtrain=dtr,
            num_boost_round=10000,
            evals=[(dva, "val")],
            early_stopping_rounds=200,
            verbose_eval=False,
        )
        # вернуть "модель" с привычными методами
        return XGBBoosterWrapper(booster, xgb_params)

    if get_params:
        results = pd.DataFrame(results).sort_values("cv_auc", ascending=False)
        return results

    return best_params


def oof_pred(df: pd.DataFrame, y: pd.Series, model_name: str, params: Dict[str, Any], cv=5, random_state=RANDOM_SEED) -> Tuple[np.ndarray, float]:
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    aucs = []
    preds = np.zeros(len(y))
    cat_features = df.select_dtypes(include=["category"]).columns.tolist()
    for train_idx, val_idx in tqdm(skf.split(df, y), total=skf.get_n_splits(), desc=f"OOF {model_name}"):
        df_train, df_val = df.iloc[train_idx], df.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if model_name == "xgb":
            model = XGBClassifier(**params)
            model.fit(df_train, y_train)
        elif model_name == "lgbm":
            model = LGBMClassifier(**params)
            model.fit(df_train, y_train, categorical_feature=cat_features)
        elif model_name == "cb":
            train_pool = Pool(df_train, y_train, cat_features=cat_features)
            model = CatBoostClassifier(**params)
            model.fit(train_pool)
        else:
            raise ValueError("name must be xgb, cb or lgbm")

        fold_pred = model.predict_proba(df_val)[:, 1]
        preds[val_idx] = fold_pred
        auc = roc_auc_score(y_val, fold_pred)
        aucs.append(auc)
    return preds, np.mean(aucs)

def test_pred_syn_train(df: pd.DataFrame, y: pd.Series, df_test : pd.DataFrame, model_name: str, params: Dict[str, Any]) -> np.ndarray:
    cat_features = df.select_dtypes(include=["category"]).columns.tolist()
    if model_name == "xgb":
        model = XGBClassifier(**params)
        model.fit(df, y)
    elif model_name == "lgbm":
        model = LGBMClassifier(**params)
        model.fit(df, y, categorical_feature=cat_features)
    elif model_name == "cb":
        train_pool = Pool(df, y, cat_features=cat_features)
        model = CatBoostClassifier(**params)
        model.fit(train_pool)
    else:
        raise ValueError("name must be xgb, cb or lgbm")

    pred = model.predict_proba(df_test)[:, 1]
    return pred

