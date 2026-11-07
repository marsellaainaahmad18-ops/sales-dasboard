
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


METHOD_NAME = "K-Medoids + DTW"
CLUSTER_COL = "KMedoids_Cluster"


def _init_medoids_plus_plus(
    distance_matrix,
    n_clusters,
    random_state=42,
):
    n_samples = distance_matrix.shape[0]
    rng = np.random.default_rng(random_state)

    medoids = [
        int(rng.integers(0, n_samples))
    ]

    while len(medoids) < n_clusters:
        nearest_distance = np.min(
            distance_matrix[:, medoids],
            axis=1,
        )

        weights = nearest_distance ** 2
        weights[medoids] = 0.0

        if weights.sum() <= 0:
            remaining = np.setdiff1d(
                np.arange(n_samples),
                np.asarray(medoids),
            )
            next_medoid = int(
                rng.choice(remaining)
            )
        else:
            p = weights / weights.sum()
            next_medoid = int(
                rng.choice(
                    n_samples,
                    p=p,
                )
            )

        if next_medoid not in medoids:
            medoids.append(next_medoid)

    return np.asarray(medoids, dtype=int)


def _single_run(
    D,
    n_clusters,
    max_iter=100,
    random_state=42,
):
    medoids = _init_medoids_plus_plus(
        D,
        n_clusters,
        random_state,
    )

    cost_history = []

    for iteration in range(max_iter):
        distance_to_medoids = D[:, medoids]
        labels = np.argmin(distance_to_medoids, axis=1)

        cost_history.append(
            float(
                np.min(
                    distance_to_medoids,
                    axis=1,
                ).sum()
            )
        )

        new_medoids = medoids.copy()

        for cluster_id in range(n_clusters):
            members = np.where(
                labels == cluster_id
            )[0]

            if len(members) == 0:
                nearest = np.min(
                    D[:, medoids],
                    axis=1,
                )
                nearest[medoids] = -np.inf
                new_medoids[cluster_id] = int(
                    np.argmax(nearest)
                )
                continue

            within = D[
                np.ix_(
                    members,
                    members,
                )
            ]

            total_distance = within.sum(axis=1)

            new_medoids[cluster_id] = int(
                members[
                    np.argmin(total_distance)
                ]
            )

        if np.array_equal(
            np.sort(new_medoids),
            np.sort(medoids),
        ):
            medoids = new_medoids
            break

        medoids = new_medoids

    distance_to_medoids = D[:, medoids]
    labels = np.argmin(distance_to_medoids, axis=1)
    nearest_distance = np.min(distance_to_medoids, axis=1)

    return {
        "medoids": medoids,
        "labels": labels,
        "distance_to_medoid": nearest_distance,
        "total_cost": float(nearest_distance.sum()),
        "average_distance": float(nearest_distance.mean()),
        "n_iter": iteration + 1,
        "cost_history": cost_history,
    }


def _fit_kmedoids(
    D,
    n_clusters,
    n_init=5,
    max_iter=100,
    random_state=42,
):
    best = None

    for run_idx in range(n_init):
        result = _single_run(
            D,
            n_clusters,
            max_iter=max_iter,
            random_state=random_state + run_idx,
        )

        if (
            best is None
            or result["total_cost"]
            < best["total_cost"]
        ):
            best = result

    return best


def run(df: pd.DataFrame) -> dict:
    X = get_normalized_matrix(df)
    D = get_dtw_matrix(X)

    rows = []
    candidates = {}

    for k in range(2, 7):
        result = _fit_kmedoids(
            D,
            n_clusters=k,
            n_init=5,
            max_iter=100,
            random_state=42,
        )

        labels = result["labels"]

        sil = silhouette_score(
            D,
            labels,
            metric="precomputed",
        )

        counts = np.bincount(
            labels,
            minlength=k,
        )

        rows.append({
            "k": k,
            "Silhouette_DTW": sil,
            "Total_Cost": result["total_cost"],
            "Average_Distance_to_Medoid":
                result["average_distance"],
            "Smallest_Cluster": int(counts.min()),
            "Largest_Cluster": int(counts.max()),
            "Iterations": result["n_iter"],
        })

        candidates[k] = result

    evaluation = pd.DataFrame(rows)

    valid = (
        evaluation
        .sort_values(
            ["Silhouette_DTW", "Total_Cost"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    best_row = valid.iloc[0]
    best_k = int(best_row["k"])
    final = candidates[best_k]

    result_df = attach_labels(
        df,
        final["labels"],
        CLUSTER_COL,
    )

    result_df["Distance_to_Medoid"] = (
        final["distance_to_medoid"]
    )

    medoid_table = pd.DataFrame({
        "Cluster": np.arange(best_k),
        "Medoid_Index": final["medoids"],
        "Medoid_Product": [
            str(
                result_df.iloc[i]["Product_Code"]
            )
            for i in final["medoids"]
        ],
    })

    return {
        "method": METHOD_NAME,
        "cluster_col": CLUSTER_COL,
        "labels": final["labels"],
        "result_df": result_df,
        "evaluation_df": evaluation,
        "summary_df": cluster_summary(result_df, CLUSTER_COL),
        "pattern_df": mean_pattern_long(result_df, CLUSTER_COL),
        "best_params": {"k": best_k},
        "primary_metric_name": "Silhouette DTW",
        "primary_metric_value": float(best_row["Silhouette_DTW"]),
        "secondary_metric_name": "Total Cost",
        "secondary_metric_value": float(best_row["Total_Cost"]),
        "medoid_df": medoid_table,
        "cost_history": final["cost_history"],
        "notes": (
            "Sumber hitung metode adalah methods/kmedoids_dtw.py. "
            "Dashboard tidak menyimpan ulang algoritmanya."
        ),
    }
