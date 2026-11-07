
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from .common import (
    attach_labels,
    cluster_summary,
    get_normalized_matrix,
    mean_pattern_long,
)


METHOD_NAME = "Fuzzy C-Means"
CLUSTER_COL = "FCM_Cluster"


def _single(
    X,
    c,
    m=2.0,
    error=1e-5,
    max_iter=500,
    random_state=42,
):
    n = X.shape[0]
    rng = np.random.default_rng(random_state)
    eps = 1e-12

    U = rng.random((c, n))
    U = U / U.sum(axis=0, keepdims=True)

    objective_history = []

    for iteration in range(max_iter):
        U_old = U.copy()
        U_m = U ** m

        centers = (
            U_m @ X
        ) / (
            U_m.sum(
                axis=1,
                keepdims=True,
            ) + eps
        )

        diff = (
            centers[:, None, :]
            - X[None, :, :]
        )

        distances = np.sqrt(
            np.sum(diff ** 2, axis=2)
        )
        distances = np.maximum(
            distances,
            eps,
        )

        objective = np.sum(
            (U ** m)
            * (distances ** 2)
        )
        objective_history.append(
            float(objective)
        )

        power = 2.0 / (m - 1.0)

        ratio = (
            distances[:, None, :]
            / distances[None, :, :]
        ) ** power

        U = 1.0 / ratio.sum(axis=1)

        if np.max(
            np.abs(U - U_old)
        ) < error:
            break

    U_m = U ** m

    centers = (
        U_m @ X
    ) / (
        U_m.sum(
            axis=1,
            keepdims=True,
        ) + eps
    )

    diff = (
        centers[:, None, :]
        - X[None, :, :]
    )
    distances = np.sqrt(
        np.sum(diff ** 2, axis=2)
    )

    objective = float(
        np.sum(
            (U ** m)
            * (distances ** 2)
        )
    )

    return {
        "centers": centers,
        "membership": U,
        "objective": objective,
        "n_iter": iteration + 1,
        "objective_history": objective_history,
    }


def _fit(
    X,
    c,
    n_init=5,
):
    best = None

    for run_idx in range(n_init):
        candidate = _single(
            X,
            c,
            m=2.0,
            error=1e-5,
            max_iter=500,
            random_state=42 + run_idx,
        )

        if (
            best is None
            or candidate["objective"]
            < best["objective"]
        ):
            best = candidate

    return best


def _xie_beni(
    X,
    centers,
    U,
    m=2.0,
):
    eps = 1e-12

    diff = (
        centers[:, None, :]
        - X[None, :, :]
    )
    distance_sq = np.sum(
        diff ** 2,
        axis=2,
    )

    numerator = np.sum(
        (U ** m)
        * distance_sq
    )

    center_diff = (
        centers[:, None, :]
        - centers[None, :, :]
    )
    center_distance_sq = np.sum(
        center_diff ** 2,
        axis=2,
    )

    np.fill_diagonal(
        center_distance_sq,
        np.inf,
    )

    min_sep = float(
        np.min(center_distance_sq)
    )

    xb = numerator / (
        X.shape[0]
        * max(min_sep, eps)
    )

    return float(xb), min_sep


def run(df: pd.DataFrame) -> dict:
    X = get_normalized_matrix(df)

    rows = []
    candidates = {}

    for c in range(2, 7):
        result = _fit(
            X,
            c,
            n_init=5,
        )

        U = result["membership"]
        centers = result["centers"]
        labels = np.argmax(U, axis=0)

        counts = np.bincount(
            labels,
            minlength=c,
        )

        fpc = float(
            np.sum(U ** 2)
            / U.shape[1]
        )

        entropy = float(
            -np.sum(
                U * np.log(U + 1e-12)
            )
            / U.shape[1]
        )

        xb, min_sep = _xie_beni(
            X,
            centers,
            U,
        )

        n_formed = len(
            np.unique(labels)
        )

        sil = (
            silhouette_score(
                X,
                labels,
                metric="euclidean",
            )
            if n_formed == c
            else np.nan
        )

        rows.append({
            "c": c,
            "Objective": result["objective"],
            "FPC": fpc,
            "Partition_Entropy": entropy,
            "Xie_Beni": xb,
            "Silhouette": sil,
            "Hard_Clusters_Formed": n_formed,
            "Min_Center_Separation_Sq": min_sep,
            "Smallest_Hard_Cluster": int(counts.min()),
            "Iterations": result["n_iter"],
        })

        candidates[c] = result

    evaluation = pd.DataFrame(rows)

    valid = evaluation[
        (
            evaluation["Hard_Clusters_Formed"]
            == evaluation["c"]
        )
        & (
            evaluation[
                "Min_Center_Separation_Sq"
            ] > 1e-8
        )
        & np.isfinite(
            evaluation["Xie_Beni"]
        )
    ].copy()

    valid = (
        valid
        .sort_values(
            [
                "Xie_Beni",
                "FPC",
                "Silhouette",
            ],
            ascending=[
                True,
                False,
                False,
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    if valid.empty:
        raise RuntimeError(
            "Tidak ada kandidat FCM yang valid."
        )

    best_row = valid.iloc[0]
    best_c = int(best_row["c"])

    final = candidates[best_c]
    U = final["membership"]
    labels = np.argmax(U, axis=0)

    result_df = attach_labels(
        df,
        labels,
        CLUSTER_COL,
    )

    dominant = np.max(U, axis=0)
    sorted_U = np.sort(U, axis=0)
    second = sorted_U[-2]
    margin = dominant - second

    membership_entropy = (
        -np.sum(
            U * np.log(U + 1e-12),
            axis=0,
        )
        / np.log(best_c)
    )

    result_df["Dominant_Membership"] = dominant
    result_df["Second_Membership"] = second
    result_df["Membership_Margin"] = margin
    result_df["Membership_Entropy"] = membership_entropy

    for j in range(best_c):
        result_df[
            f"Membership_Cluster_{j}"
        ] = U[j]

    return {
        "method": METHOD_NAME,
        "cluster_col": CLUSTER_COL,
        "labels": labels,
        "result_df": result_df,
        "evaluation_df": evaluation,
        "summary_df": cluster_summary(result_df, CLUSTER_COL),
        "pattern_df": mean_pattern_long(result_df, CLUSTER_COL),
        "best_params": {
            "c": best_c,
            "m": 2.0,
        },
        "primary_metric_name": "Xie-Beni",
        "primary_metric_value": float(best_row["Xie_Beni"]),
        "secondary_metric_name": "FPC",
        "secondary_metric_value": float(best_row["FPC"]),
        "centers": final["centers"],
        "objective_history": final["objective_history"],
        "notes": (
            "FCM dijalankan dari methods/fuzzy_cmeans.py. "
            "Dashboard dan notebook connected memanggil file yang sama."
        ),
    }
