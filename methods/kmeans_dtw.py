
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import (
    attach_labels,
    cluster_summary,
    get_normalized_matrix,
    mean_pattern_long,
)


METHOD_NAME = "K-Means + DTW"
CLUSTER_COL = "KMeans_Cluster"


def run(df: pd.DataFrame) -> dict:
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.clustering import silhouette_score as ts_silhouette_score

    X = get_normalized_matrix(df)
    X_ts = X[:, :, None]

    rows = []
    sample_size = min(300, len(X))

    for k in range(2, 7):
        model = TimeSeriesKMeans(
            n_clusters=k,
            metric="dtw",
            n_init=2,
            max_iter=20,
            max_iter_barycenter=10,
            random_state=42,
            n_jobs=-1,
            verbose=False,
        )

        labels = model.fit_predict(X_ts)

        sil = ts_silhouette_score(
            X_ts,
            labels,
            metric="dtw",
            sample_size=sample_size,
            random_state=42,
            n_jobs=-1,
        )

        rows.append({
            "k": k,
            "Inertia_DTW": float(model.inertia_),
            "Silhouette_DTW": float(sil),
        })

    evaluation = pd.DataFrame(rows)

    best_row = evaluation.loc[
        evaluation["Silhouette_DTW"].idxmax()
    ]

    best_k = int(best_row["k"])

    final_model = TimeSeriesKMeans(
        n_clusters=best_k,
        metric="dtw",
        n_init=5,
        max_iter=50,
        max_iter_barycenter=20,
        random_state=42,
        n_jobs=-1,
        verbose=False,
    )

    labels = final_model.fit_predict(X_ts)

    result_df = attach_labels(
        df,
        labels,
        CLUSTER_COL,
    )

    return {
        "method": METHOD_NAME,
        "cluster_col": CLUSTER_COL,
        "labels": labels,
        "result_df": result_df,
        "evaluation_df": evaluation,
        "summary_df": cluster_summary(result_df, CLUSTER_COL),
        "pattern_df": mean_pattern_long(result_df, CLUSTER_COL),
        "best_params": {"k": best_k},
        "primary_metric_name": "Silhouette DTW",
        "primary_metric_value": float(best_row["Silhouette_DTW"]),
        "secondary_metric_name": "Inertia DTW",
        "secondary_metric_value": float(
            evaluation.loc[
                evaluation["k"] == best_k,
                "Inertia_DTW",
            ].iloc[0]
        ),
        "centers": final_model.cluster_centers_[:, :, 0],
        "notes": (
            "Tuning dan final fit mengikuti notebook K-Means: "
            "k=2-6, Silhouette DTW sebagai utama, random_state=42."
        ),
    }
