import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score
import gower
from sklearn.utils import check_random_state


def make_mixed_matrix(df: pd.DataFrame, num_cols, cat_cols):
    X_num = df[num_cols].to_numpy(dtype=float)
    X_cat = df[cat_cols].astype(str).to_numpy()
    X_mix = np.concatenate([X_num, X_cat], axis=1)
    cat_idx = list(range(X_num.shape[1], X_mix.shape[1]))
    return X_mix, cat_idx


def tune_kproto_fast(df_train, num_cols, cat_cols,
                     K_list, gamma_list,
                     n_sample_cost=40000, n_sample_sil=8000,
                     n_init=3, max_iter=50, seed=42):
    rng = check_random_state(seed)

    if len(df_train) > n_sample_cost:
        train_cost = df_train.sample(n=n_sample_cost, random_state=seed)
    else:
        train_cost = df_train

    X_cost, cat_idx_cost = make_mixed_matrix(train_cost, num_cols, cat_cols)

    results = []
    for K in K_list:
        for gamma in gamma_list:
            kp = KPrototypes(
                n_clusters=K, gamma=gamma,
                init='Cao', n_init=n_init, max_iter=max_iter,
                random_state=seed, verbose=0
            )
            labels_cost = kp.fit_predict(X_cost, categorical=cat_idx_cost)
            cost = kp.cost_

            if len(train_cost) > n_sample_sil:
                subs = train_cost.sample(n=n_sample_sil, random_state=seed)
            else:
                subs = train_cost


            gower_input = pd.concat(
                [subs[num_cols], subs[cat_cols].astype(str)],
                axis=1
            )
            D = gower.gower_matrix(gower_input)

            X_subs, cat_idx_subs = make_mixed_matrix(subs, num_cols, cat_cols)
            labs_sub = kp.predict(X_subs, categorical=cat_idx_subs)

            sil = silhouette_score(D, labs_sub, metric='precomputed')
            results.append({'K': K, 'gamma': gamma, 'cost': cost, 'silhouette': sil})

    res = pd.DataFrame(results).sort_values(['silhouette', 'cost'], ascending=[False, True])
    return res


def fit_final_kproto(df_train, df_val, num_cols, cat_cols, K, gamma, seed=42):
    Xtr, cat_idx = make_mixed_matrix(df_train, num_cols, cat_cols)
    Xva, _ = make_mixed_matrix(df_val, num_cols, cat_cols)

    kp = KPrototypes(
        n_clusters=K, gamma=gamma,
        init='Huang', n_init=5, max_iter=100,
        random_state=seed, verbose=1
    )
    labels_tr = kp.fit_predict(Xtr, categorical=cat_idx)
    labels_va = kp.predict(Xva, categorical=cat_idx)
    return kp, labels_tr, labels_va


def cluster_profiles(kp: KPrototypes, num_cols, cat_cols):
    centers_num, centers_cat = kp.cluster_centroids_
    rows = []
    for k in range(kp.n_clusters):
        row = {'cluster': k}
        for i, col in enumerate(num_cols):
            row[f'mean_{col}'] = centers_num[k, i]
        for j, col in enumerate(cat_cols):
            row[f'mode_{col}'] = centers_cat[k, j]
        rows.append(row)
    return pd.DataFrame(rows).sort_values('cluster')



