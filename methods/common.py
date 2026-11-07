
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

WEEK_COLS = [f"W{i}" for i in range(52)]
NORM_COLS = [f"Normalized {i}" for i in range(52)]

_DTW_CACHE: Dict[str, np.ndarray] = {}
_SBD_CACHE: Dict[str, np.ndarray] = {}


def validate_input(df: pd.DataFrame) -> None:
    required = ["Product_Code"] + WEEK_COLS
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "Kolom data utama tidak lengkap. "
            f"Kolom yang belum ada: {missing[:12]}"
        )

    if df["Product_Code"].isna().any():
        raise ValueError("Product_Code mengandung missing value.")

    if df["Product_Code"].duplicated().any():
        raise ValueError(
            "Product_Code harus unik agar hasil per metode bisa "
            "dihubungkan ke produk yang sama."
        )

    if df[WEEK_COLS].isna().any().any():
        raise ValueError("W0-W51 mengandung missing value.")


def has_normalized_columns(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in NORM_COLS)


def get_raw_matrix(df: pd.DataFrame) -> np.ndarray:
    validate_input(df)
    return df[WEEK_COLS].to_numpy(dtype=float)


def get_normalized_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Gunakan Normalized 0-51 bila tersedia, sesuai notebook.
    Bila tidak tersedia, fallback min-max per produk agar dashboard
    tetap bisa berjalan dari W0-W51 saja.
    """
    validate_input(df)

    if has_normalized_columns(df):
        X = df[NORM_COLS].to_numpy(dtype=float)
        if np.isnan(X).any():
            raise ValueError("Kolom Normalized 0-51 mengandung NaN.")
        return X

    X = df[WEEK_COLS].to_numpy(dtype=float)
    row_min = X.min(axis=1, keepdims=True)
    row_max = X.max(axis=1, keepdims=True)
    denom = row_max - row_min

    X_norm = np.divide(
        X - row_min,
        denom,
        out=np.zeros_like(X, dtype=float),
        where=denom > 1e-12,
    )
    return X_norm


def data_signature(X: np.ndarray) -> str:
    arr = np.ascontiguousarray(X, dtype=np.float64)
    h = hashlib.sha1()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def get_dtw_matrix(X: np.ndarray) -> np.ndarray:
    """
    Matriks DTW bersama untuk K-Medoids dan HDBSCAN.
    Cache berada di module common.py, sehingga metode DTW memakai
    sumber jarak yang sama dalam satu proses Streamlit.
    """
    key = data_signature(X)

    if key in _DTW_CACHE:
        return _DTW_CACHE[key]

    from tslearn.metrics import cdist_dtw

    D = cdist_dtw(
        X,
        n_jobs=-1,
        verbose=0,
    )

    D = np.asarray(D, dtype=float)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)

    _DTW_CACHE[key] = D
    return D


def sbd_distance_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)

    if norm_x < 1e-12 or norm_y < 1e-12:
        return np.nan

    cc = np.correlate(x, y, mode="full")
    ncc = cc / (norm_x * norm_y)

    return float(
        np.clip(
            1.0 - np.max(ncc),
            0.0,
            2.0,
        )
    )


def get_sbd_matrix(X_2d: np.ndarray) -> np.ndarray:
    key = data_signature(X_2d)

    if key in _SBD_CACHE:
        return _SBD_CACHE[key]

    n = X_2d.shape[0]
    D = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            d = sbd_distance_1d(
                X_2d[i],
                X_2d[j],
            )
            D[i, j] = d
            D[j, i] = d

    _SBD_CACHE[key] = D
    return D


def add_sales_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["Total_Sales"] = out[WEEK_COLS].sum(axis=1)
    out["Mean_Weekly_Sales"] = out[WEEK_COLS].mean(axis=1)
    out["Std_Weekly_Sales"] = out[WEEK_COLS].std(axis=1)
    out["Max_Weekly_Sales"] = out[WEEK_COLS].max(axis=1)
    out["Zero_Weeks"] = (out[WEEK_COLS] == 0).sum(axis=1)

    return out


def attach_labels(
    df: pd.DataFrame,
    labels: np.ndarray,
    cluster_col: str,
) -> pd.DataFrame:
    out = add_sales_features(df)

    if len(labels) != len(out):
        raise ValueError(
            f"Panjang label ({len(labels)}) tidak sama dengan "
            f"jumlah produk ({len(out)})."
        )

    out[cluster_col] = labels
    return out


def cluster_summary(
    result_df: pd.DataFrame,
    cluster_col: str,
) -> pd.DataFrame:
    valid = result_df.copy()
    valid["_cluster_num"] = pd.to_numeric(
        valid[cluster_col],
        errors="coerce",
    )

    valid = valid[
        valid["_cluster_num"] >= 0
    ].copy()

    if valid.empty:
        return pd.DataFrame()

    summary = (
        valid
        .groupby("_cluster_num")
        .agg(
            Jumlah_Produk=("Product_Code", "count"),
            Rata2_Total_Sales=("Total_Sales", "mean"),
            Median_Total_Sales=("Total_Sales", "median"),
            Rata2_Penjualan_Mingguan=("Mean_Weekly_Sales", "mean"),
            Rata2_Maksimum_Mingguan=("Max_Weekly_Sales", "mean"),
            Rata2_Minggu_Nol=("Zero_Weeks", "mean"),
        )
        .reset_index()
        .rename(columns={"_cluster_num": "Cluster"})
    )

    return summary


def mean_pattern_long(
    result_df: pd.DataFrame,
    cluster_col: str,
) -> pd.DataFrame:
    temp = result_df.copy()
    temp["_cluster_num"] = pd.to_numeric(
        temp[cluster_col],
        errors="coerce",
    )

    temp = temp[
        temp["_cluster_num"] >= 0
    ]

    rows = []

    for cluster, group in temp.groupby("_cluster_num"):
        means = group[WEEK_COLS].mean(axis=0).to_numpy(dtype=float)

        for week, value in enumerate(means):
            rows.append({
                "Cluster": f"Cluster {int(cluster)}",
                "Minggu": week,
                "Rata-rata Penjualan": float(value),
            })

    return pd.DataFrame(rows)
