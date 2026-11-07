
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from .common import (
    attach_labels,
    cluster_summary,
    get_dtw_matrix,
    get_normalized_matrix,
    mean_pattern_long,
)


METHOD_NAME = "HDBSCAN + DTW"
CLUSTER_COL = "HDBSCAN_Cluster"


def run(df: pd.DataFrame) -> dict:
    import hdbscan
    from hdbscan.validity import validity_index

    X = get_normalized_matrix(df)
    D = get_dtw_matrix(X)

    rows = []
    models = {}

    for min_cluster_size in [15, 25, 40, 60]:
        for min_samples in [5, 10, 15]:
            model = hdbscan.HDBSCAN(
                metric="precomputed",
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_method="eom",
            )

            labels = model.fit_predict(D)

            non_noise = labels != -1
            non_noise_labels = labels[non_noise]
            n_clusters = len(
                set(non_noise_labels.tolist())
            )

            n_noise = int(
                np.sum(labels == -1)
            )
            noise_pct = (
                100.0
                * n_noise
                / len(labels)
            )

            sil = np.nan

            if (
                n_clusters >= 2
                and non_noise.sum() > n_clusters
            ):
                sub_D = D[
                    np.ix_(
                        non_noise,
                        non_noise,
                    )
                ]

                sil = silhouette_score(
                    sub_D,
                    non_noise_labels,
                    metric="precomputed",
                )

            dbcv = np.nan

            if n_clusters >= 2:
                try:
                    dbcv = validity_index(
                        D,
                        labels,
                        metric="precomputed",
                        d=X.shape[1],
                    )
                except Exception:
                    dbcv = np.nan

            persistence = (
                float(
                    np.mean(
                        model.cluster_persistence_
                    )
                )
                if len(model.cluster_persistence_) > 0
                else np.nan
            )

            rows.append({
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_pct": noise_pct,
                "DBCV": dbcv,
                "silhouette_non_noise": sil,
                "mean_cluster_persistence": persistence,
            })

            models[
                (
                    min_cluster_size,
                    min_samples,
                )
            ] = model

    evaluation = pd.DataFrame(rows)

    valid = evaluation[
        (evaluation["n_clusters"] >= 2)
        & evaluation["DBCV"].notna()
    ].copy()

    if valid.empty:
        raise RuntimeError(
            "Tidak ada kombinasi HDBSCAN yang menghasilkan "
            "minimal 2 cluster dengan DBCV valid."
        )

    preferred = valid[
        valid["noise_pct"] <= 50
    ].copy()

    pool = (
        preferred
        if not preferred.empty
        else valid
    )

    pool = (
        pool
        .sort_values(
            [
                "DBCV",
                "silhouette_non_noise",
                "noise_pct",
            ],
            ascending=[
                False,
                False,
                True,
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    best_row = pool.iloc[0]

    best_mcs = int(
        best_row["min_cluster_size"]
    )
    best_ms = int(
        best_row["min_samples"]
    )

    final_model = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=best_mcs,
        min_samples=best_ms,
        cluster_selection_method="eom",
    )

    labels = final_model.fit_predict(D)

    result_df = attach_labels(
        df,
        labels,
        CLUSTER_COL,
    )

    result_df["Cluster_Probability"] = (
        final_model.probabilities_
    )
    result_df["Outlier_Score"] = (
        final_model.outlier_scores_
    )

    return {
        "method": METHOD_NAME,
        "cluster_col": CLUSTER_COL,
        "labels": labels,
        "result_df": result_df,
        "evaluation_df": evaluation,
        "summary_df": cluster_summary(result_df, CLUSTER_COL),
        "pattern_df": mean_pattern_long(result_df, CLUSTER_COL),
        "best_params": {
            "min_cluster_size": best_mcs,
            "min_samples": best_ms,
        },
        "primary_metric_name": "DBCV",
        "primary_metric_value": float(best_row["DBCV"]),
        "secondary_metric_name": "Noise %",
        "secondary_metric_value": float(best_row["noise_pct"]),
        "special": {
            "noise_count": int(np.sum(labels == -1)),
            "noise_pct": float(
                100.0
                * np.sum(labels == -1)
                / len(labels)
            ),
        },
        "notes": (
            "HDBSCAN benar-benar dijalankan dari "
            "methods/hdbscan_dtw.py dengan grid parameter notebook."
        ),
    }
