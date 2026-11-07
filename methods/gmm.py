
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from .common import (
    attach_labels,
    cluster_summary,
    get_normalized_matrix,
    mean_pattern_long,
)


METHOD_NAME = "Gaussian Mixture Model"
CLUSTER_COL = "GMM_Cluster"


def _fit_model(
    X,
    n_components,
    covariance_type,
):
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        n_init=3,
        max_iter=500,
        tol=1e-4,
        reg_covar=1e-5,
        random_state=42,
        init_params="kmeans",
    )

    model.fit(X)

    labels = model.predict(X)
    probabilities = model.predict_proba(X)

    counts = np.bincount(
        labels,
        minlength=n_components,
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
        if n_formed >= 2
        else np.nan
    )

    row = {
        "n_components": n_components,
        "covariance_type": covariance_type,
        "BIC": model.bic(X),
        "AIC": model.aic(X),
        "Average_Log_Likelihood": model.score(X),
        "Silhouette": sil,
        "Converged": model.converged_,
        "Iterations": model.n_iter_,
        "Clusters_Formed": n_formed,
        "Smallest_Cluster": int(counts.min()),
        "Largest_Cluster": int(counts.max()),
        "Mean_Max_Probability": float(
            probabilities.max(axis=1).mean()
        ),
    }

    return model, row


def run(df: pd.DataFrame) -> dict:
    X = get_normalized_matrix(df)

    rows = []
    candidates = {}

    covariance_types = [
        "spherical",
        "diag",
        "tied",
        "full",
    ]

    for covariance in covariance_types:
        for k in range(2, 7):
            model, row = _fit_model(
                X,
                k,
                covariance,
            )

            rows.append(row)
            candidates[(k, covariance)] = model

    initial_df = pd.DataFrame(rows)

    valid_initial = initial_df[
        (initial_df["Converged"] == True)
        & (
            initial_df["Clusters_Formed"]
            == initial_df["n_components"]
        )
    ].copy()

    if valid_initial.empty:
        raise RuntimeError(
            "Tidak ada kandidat GMM awal yang valid."
        )

    best_initial = (
        valid_initial
        .sort_values("BIC")
        .iloc[0]
    )

    best_initial_k = int(
        best_initial["n_components"]
    )
    best_initial_cov = str(
        best_initial["covariance_type"]
    )

    if best_initial_k == 6:
        for k in range(7, 11):
            model, row = _fit_model(
                X,
                k,
                best_initial_cov,
            )

            rows.append(row)
            candidates[
                (k, best_initial_cov)
            ] = model

    evaluation = pd.DataFrame(rows)

    valid = evaluation[
        (evaluation["Converged"] == True)
        & (
            evaluation["Clusters_Formed"]
            == evaluation["n_components"]
        )
    ].copy()

    valid = (
        valid
        .sort_values(
            ["BIC", "AIC", "Silhouette"],
            ascending=[True, True, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )

    if valid.empty:
        raise RuntimeError(
            "Tidak ada model GMM yang valid."
        )

    best_row = valid.iloc[0]

    best_k = int(
        best_row["n_components"]
    )
    best_cov = str(
        best_row["covariance_type"]
    )

    final_model = candidates[
        (best_k, best_cov)
    ]

    labels = final_model.predict(X)
    probabilities = final_model.predict_proba(X)

    result_df = attach_labels(
        df,
        labels,
        CLUSTER_COL,
    )

    dominant = probabilities.max(axis=1)
    sorted_prob = np.sort(
        probabilities,
        axis=1,
    )
    second = sorted_prob[:, -2]
    margin = dominant - second

    entropy = (
        -np.sum(
            probabilities
            * np.log(
                probabilities + 1e-12
            ),
            axis=1,
        )
        / np.log(best_k)
    )

    result_df["Dominant_Probability"] = dominant
    result_df["Second_Probability"] = second
    result_df["Probability_Margin"] = margin
    result_df["Probability_Entropy"] = entropy

    for j in range(best_k):
        result_df[
            f"Probability_Cluster_{j}"
        ] = probabilities[:, j]

    return {
        "method": METHOD_NAME,
        "cluster_col": CLUSTER_COL,
        "labels": labels,
        "result_df": result_df,
        "evaluation_df": evaluation,
        "summary_df": cluster_summary(result_df, CLUSTER_COL),
        "pattern_df": mean_pattern_long(result_df, CLUSTER_COL),
        "best_params": {
            "n_components": best_k,
            "covariance_type": best_cov,
        },
        "primary_metric_name": "BIC",
        "primary_metric_value": float(best_row["BIC"]),
        "secondary_metric_name": "AIC",
        "secondary_metric_value": float(best_row["AIC"]),
        "centers": final_model.means_,
        "mixture_weights": final_model.weights_,
        "notes": (
            "GMM dijalankan langsung dari methods/gmm.py. "
            "Jika file ini diubah, dashboard dan notebook connected ikut berubah."
        ),
    }
