
from __future__ import annotations

import os

# Stabilitas native thread pada macOS/Jupyter
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from .common import (
    attach_labels,
    cluster_summary,
    get_raw_matrix,
    get_sbd_matrix,
    mean_pattern_long,
)


METHOD_NAME = "k-Shape"
CLUSTER_COL = "KShape_Cluster"


def run(df: pd.DataFrame) -> dict:
    from tslearn.clustering import KShape
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance

    X_raw = get_raw_matrix(df)

    std = X_raw.std(axis=1)
    constant_mask = std < 1e-12
    model_mask = ~constant_mask

    model_indices = np.where(
        model_mask
    )[0]

    X_model = X_raw[
        model_mask
    ]

    if len(X_model) < 3:
        raise RuntimeError(
            "Produk non-konstan terlalu sedikit untuk k-Shape."
        )

    X_ts = X_model[:, :, None]

    scaler = TimeSeriesScalerMeanVariance(
        mu=0.0,
        std=1.0,
    )

    X_scaled = scaler.fit_transform(X_ts)
    X_2d = X_scaled[:, :, 0]

    D = get_sbd_matrix(X_2d)

    rows = []
    candidates = {}

    for k in range(2, 7):
        try:
            model = KShape(
                n_clusters=k,
                n_init=3,
                max_iter=100,
                tol=1e-6,
                random_state=42,
                verbose=False,
            )

            labels = model.fit_predict(
                X_scaled
            )

            n_formed = len(
                np.unique(labels)
            )

            counts = np.bincount(
                labels,
                minlength=k,
            )

            sil = (
                silhouette_score(
                    D,
                    labels,
                    metric="precomputed",
                )
                if n_formed >= 2
                else np.nan
            )

            rows.append({
                "k": k,
                "Silhouette_SBD": sil,
                "Inertia_KShape": float(model.inertia_),
                "Clusters_Formed": n_formed,
                "Smallest_Cluster": int(counts.min()),
                "Largest_Cluster": int(counts.max()),
                "Iterations": int(model.n_iter_),
            })

            candidates[k] = {
                "model": model,
                "labels": labels,
            }

        except Exception:
            rows.append({
                "k": k,
                "Silhouette_SBD": np.nan,
                "Inertia_KShape": np.nan,
                "Clusters_Formed": 0,
                "Smallest_Cluster": 0,
                "Largest_Cluster": 0,
                "Iterations": 0,
            })

    evaluation = pd.DataFrame(rows)

    valid = evaluation[
        (
            evaluation["Clusters_Formed"]
            == evaluation["k"]
        )
        & evaluation[
            "Silhouette_SBD"
        ].notna()
    ].copy()

    if valid.empty:
        raise RuntimeError(
            "Tidak ada kandidat k-Shape yang valid."
        )

    valid = (
        valid
        .sort_values(
            [
                "Silhouette_SBD",
                "Inertia_KShape",
                "Smallest_Cluster",
            ],
            ascending=[
                False,
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    best_row = valid.iloc[0]
    best_k = int(best_row["k"])

    final = candidates[best_k]
    local_labels = final["labels"]

    labels = np.full(
        len(df),
        -1,
        dtype=int,
    )
    labels[
        model_indices
    ] = local_labels

    result_df = attach_labels(
        df,
        labels,
        CLUSTER_COL,
    )

    status = np.full(
        len(df),
        "Constant profile - not clustered",
        dtype=object,
    )
    status[
        model_indices
    ] = "Clustered"

    result_df["KShape_Status"] = status

    return {
        "method": METHOD_NAME,
        "cluster_col": CLUSTER_COL,
        "labels": labels,
        "result_df": result_df,
        "evaluation_df": evaluation,
        "summary_df": cluster_summary(result_df, CLUSTER_COL),
        "pattern_df": mean_pattern_long(result_df, CLUSTER_COL),
        "best_params": {"k": best_k},
        "primary_metric_name": "Silhouette SBD",
        "primary_metric_value": float(best_row["Silhouette_SBD"]),
        "secondary_metric_name": "Inertia k-Shape",
        "secondary_metric_value": float(best_row["Inertia_KShape"]),
        "centers": final["model"].cluster_centers_[:, :, 0],
        "special": {
            "constant_profiles": int(
                constant_mask.sum()
            )
        },
        "notes": (
            "k-Shape dijalankan dari methods/kshape.py. "
            "Produk konstan dipisahkan seperti notebook SAFE."
        ),
    }
