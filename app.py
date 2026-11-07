
from __future__ import annotations

import hashlib
import inspect
import io
from pathlib import Path
import sys

# Pastikan folder tempat app.py berada selalu dikenali Python.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import adjusted_rand_score

from methods import METHOD_RUNNERS
from methods.common import WEEK_COLS, validate_input


st.set_page_config(
    page_title="Sales Clustering Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
    .stApp { background: #F6F8FB; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg,#0F3A63,#184F7C);
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    .block-container {
        padding-top: 1.35rem;
        max-width: 1500px;
    }

    .hero {
        background: linear-gradient(120deg,#103C66,#246B9F);
        color:white;
        padding:26px 30px;
        border-radius:22px;
        margin-bottom:18px;
        box-shadow:0 10px 30px rgba(16,60,102,.12);
    }

    .hero h1 {
        color:white;
        margin:0 0 6px 0;
        font-size:2rem;
    }

    .hero p {
        color:#E7F0F7;
        margin:0;
        line-height:1.55;
    }

    .metric-card {
        background:white;
        border:1px solid #E4EAF0;
        border-radius:18px;
        padding:18px 20px;
        min-height:108px;
        box-shadow:0 5px 18px rgba(16,60,102,.05);
    }

    .metric-label {
        color:#667085;
        font-size:.82rem;
        font-weight:650;
    }

    .metric-value {
        color:#103C66;
        font-size:1.55rem;
        font-weight:800;
        margin-top:5px;
    }

    .metric-note {
        color:#788698;
        font-size:.75rem;
        margin-top:5px;
    }

    .method-title {
        font-size:1.05rem;
        font-weight:800;
        color:#103C66;
        margin-bottom:3px;
    }

    .method-sub {
        font-size:.80rem;
        color:#667085;
        margin-bottom:8px;
    }

    .info-box {
        background:#FFF8E7;
        border:1px solid #F1D48A;
        border-left:5px solid #F4B942;
        padding:14px 16px;
        border-radius:14px;
        color:#5C4A18;
        margin:8px 0 16px 0;
    }

    .connected-box {
        background:#ECF7F0;
        border:1px solid #C8E6D2;
        border-left:5px solid #21834F;
        padding:14px 16px;
        border-radius:14px;
        color:#235A39;
        margin:8px 0 16px 0;
    }


    .insight-card {
        background:white;
        border:1px solid #E4EAF0;
        border-radius:18px;
        padding:18px 20px;
        box-shadow:0 5px 18px rgba(16,60,102,.05);
        min-height:150px;
        margin-bottom:10px;
    }

    .insight-title {
        color:#103C66;
        font-size:.92rem;
        font-weight:800;
        margin-bottom:8px;
    }

    .insight-text {
        color:#344054;
        font-size:.90rem;
        line-height:1.58;
    }

    .conclusion-box {
        background:linear-gradient(120deg,#EEF6FC,#F8FBFD);
        border:1px solid #CFE0ED;
        border-left:6px solid #1E5A8C;
        border-radius:16px;
        padding:18px 20px;
        color:#24435D;
        line-height:1.62;
        margin:10px 0 18px 0;
    }

    .action-box {
        background:#FFF9EB;
        border:1px solid #F1D99B;
        border-left:6px solid #F4B942;
        border-radius:16px;
        padding:18px 20px;
        color:#5B4A1A;
        line-height:1.62;
        margin:10px 0 18px 0;
    }

    div[data-testid="stPlotlyChart"] {
        background:white;
        border:1px solid #E5EAF0;
        border-radius:18px;
        padding:8px;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 14px;
        min-height: 48px;
        font-weight: 700;
        border: 1px solid #D9E2EA;
        background: white;
        color: #103C66;
    }

    div.stButton > button:hover {
        border-color: #F4B942;
        color: #103C66;
        background: #FFF8E7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


METHOD_INFO = {
    "K-Means + DTW": ("Centroid + DTW", "K-Means"),
    "K-Medoids + DTW": ("Medoid + DTW", "K-Medoids"),
    "HDBSCAN + DTW": ("Density + DTW", "HDBSCAN"),
    "Fuzzy C-Means": ("Soft membership", "FCM"),
    "Gaussian Mixture Model": ("Probabilistic", "GMM"),
    "k-Shape": ("Shape-based", "k-Shape"),
}


def hero(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value, note=""):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """


def plot_layout(fig, height=420):
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=25, r=20, t=55, b=35),
        title_font=dict(color="#103C66"),
        font=dict(color="#263646"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#E8EDF2")
    return fig


def file_hash(file_bytes: bytes) -> str:
    return hashlib.sha1(file_bytes).hexdigest()


def result_key(file_signature: str, method_name: str) -> str:
    return f"{file_signature}::{method_name}"



def silhouette_reading(value):
    if value is None or pd.isna(value):
        return "belum dapat dinilai"
    value = float(value)
    if value >= 0.50:
        return "pemisahan cluster relatif jelas"
    if value >= 0.25:
        return "pemisahan cluster sedang"
    if value >= 0.00:
        return "cluster masih cukup berdekatan"
    return "struktur cluster kurang meyakinkan karena nilai negatif"


def cluster_profile_table(result):
    result_df = result["result_df"].copy()
    cluster_col = result["cluster_col"]

    result_df["_cluster_num"] = pd.to_numeric(
        result_df[cluster_col],
        errors="coerce",
    )

    valid_df = result_df[
        result_df["_cluster_num"] >= 0
    ].copy()

    if valid_df.empty:
        return pd.DataFrame()

    valid_df["_CV"] = np.divide(
        valid_df["Std_Weekly_Sales"].to_numpy(dtype=float),
        valid_df["Mean_Weekly_Sales"].to_numpy(dtype=float),
        out=np.zeros(len(valid_df), dtype=float),
        where=valid_df["Mean_Weekly_Sales"].to_numpy(dtype=float) > 1e-12,
    )

    profile = (
        valid_df
        .groupby("_cluster_num")
        .agg(
            Jumlah_Produk=("Product_Code", "count"),
            Rata2_Total_Sales=("Total_Sales", "mean"),
            Median_Total_Sales=("Total_Sales", "median"),
            Rata2_Penjualan_Mingguan=("Mean_Weekly_Sales", "mean"),
            Rata2_Maksimum_Mingguan=("Max_Weekly_Sales", "mean"),
            Rata2_Minggu_Nol=("Zero_Weeks", "mean"),
            Rata2_CV=("_CV", "mean"),
        )
        .reset_index()
        .rename(columns={"_cluster_num": "Cluster"})
    )

    profile["Proporsi_Produk_%"] = (
        100.0
        * profile["Jumlah_Produk"]
        / profile["Jumlah_Produk"].sum()
    )

    # Ranking relatif antar-cluster, bukan threshold absolut.
    volume_rank = profile["Rata2_Total_Sales"].rank(
        method="average",
        pct=True,
    )
    zero_rank = profile["Rata2_Minggu_Nol"].rank(
        method="average",
        pct=True,
    )
    cv_rank = profile["Rata2_CV"].rank(
        method="average",
        pct=True,
    )

    def volume_label(p):
        if p <= 1/3:
            return "Volume rendah"
        if p <= 2/3:
            return "Volume sedang"
        return "Volume tinggi"

    def activity_label(p):
        # Makin tinggi zero weeks = makin sporadis
        if p <= 1/3:
            return "Aktif"
        if p <= 2/3:
            return "Cukup aktif"
        return "Sporadis"

    def variability_label(p):
        if p <= 1/3:
            return "Relatif stabil"
        if p <= 2/3:
            return "Cukup fluktuatif"
        return "Fluktuatif"

    profile["Profil"] = [
        f"{volume_label(v)} • {activity_label(z)} • {variability_label(c)}"
        for v, z, c in zip(
            volume_rank,
            zero_rank,
            cv_rank,
        )
    ]

    return profile


def model_interpretation(selected_method, result):
    labels = np.asarray(result["labels"])
    valid = labels[labels >= 0]
    n_clusters = len(np.unique(valid))
    eval_df = result["evaluation_df"].copy()
    best_params = result["best_params"]
    metric_name = result["primary_metric_name"]
    metric_value = float(result["primary_metric_value"])

    why = ""
    meaning = ""
    quality = ""
    extra = ""

    if selected_method == "Gaussian Mixture Model":
        valid_eval = eval_df.copy()
        if "Converged" in valid_eval.columns:
            valid_eval = valid_eval[
                valid_eval["Converged"].astype(bool)
            ]
        if {
            "Clusters_Formed",
            "n_components",
        }.issubset(valid_eval.columns):
            valid_eval = valid_eval[
                valid_eval["Clusters_Formed"]
                == valid_eval["n_components"]
            ]

        valid_eval = valid_eval.sort_values("BIC")
        delta_bic = np.nan

        if len(valid_eval) >= 2:
            delta_bic = (
                float(valid_eval.iloc[1]["BIC"])
                - float(valid_eval.iloc[0]["BIC"])
            )

        k = best_params.get("n_components", n_clusters)
        cov = best_params.get("covariance_type", "—")

        why = (
            f"Model GMM yang dipilih memiliki K={k} dengan covariance='{cov}' "
            f"karena memberikan BIC terendah ({metric_value:,.2f}) "
            "di antara kandidat yang valid dan konvergen."
        )

        meaning = (
            f"Artinya 811 produk dibagi menjadi {n_clusters} kelompok probabilistik. "
            "Nomor cluster hanya label; makna bisnisnya harus dibaca dari pola penjualan "
            "masing-masing cluster."
        )

        if not pd.isna(delta_bic):
            if delta_bic >= 10:
                strength = "dukungan terhadap model terbaik sangat kuat"
            elif delta_bic >= 6:
                strength = "dukungan terhadap model terbaik kuat"
            elif delta_bic >= 2:
                strength = "model terbaik cukup lebih baik"
            else:
                strength = "model terbaik dan kandidat kedua masih sangat berdekatan"

            quality = (
                f"Selisih BIC terhadap kandidat kedua sekitar {delta_bic:,.2f}; "
                f"secara praktis {strength}. "
                "BIC tidak dinilai dari besar-kecil absolutnya, tetapi dari perbandingan antar-model."
            )
        else:
            quality = (
                "BIC harus dibaca secara relatif: model dengan BIC lebih rendah lebih disukai. "
                "Nilai absolut BIC tidak mempunyai batas 'bagus' yang universal."
            )

        if "Dominant_Probability" in result["result_df"].columns:
            p = pd.to_numeric(
                result["result_df"]["Dominant_Probability"],
                errors="coerce",
            )
            ambiguous = int((p < 0.60).sum())
            extra = (
                f"Terdapat {ambiguous} produk dengan probabilitas dominan < 0,60. "
                "Produk tersebut mempunyai keanggotaan cluster yang relatif kurang tegas."
            )

    elif selected_method == "Fuzzy C-Means":
        c = best_params.get("c", n_clusters)
        fpc = result.get("secondary_metric_value", np.nan)

        why = (
            f"Jumlah cluster dipilih c={c} karena menghasilkan Xie-Beni terendah "
            f"({metric_value:.4f}) di antara kandidat valid."
        )

        meaning = (
            f"Produk dibagi menjadi {n_clusters} cluster dominan, tetapi FCM tetap menyimpan "
            "derajat membership terhadap seluruh cluster. Jadi satu produk bisa dekat dengan "
            "lebih dari satu cluster."
        )

        quality = (
            f"FPC={float(fpc):.4f}. Xie-Beni yang lebih kecil berarti cluster relatif lebih kompak "
            "dan lebih terpisah; FPC yang lebih tinggi berarti membership lebih tegas."
            if not pd.isna(fpc)
            else "Xie-Beni dibaca relatif: semakin kecil dibanding kandidat lain, semakin baik."
        )

        if "Dominant_Membership" in result["result_df"].columns:
            m = pd.to_numeric(
                result["result_df"]["Dominant_Membership"],
                errors="coerce",
            )
            ambiguous = int((m < 0.60).sum())
            extra = (
                f"Ada {ambiguous} produk dengan dominant membership < 0,60; "
                "produk ini berada di batas antar-cluster dan perlu dibaca lebih hati-hati."
            )

    elif selected_method == "HDBSCAN + DTW":
        noise_count = int(np.sum(labels < 0))
        noise_pct = 100.0 * noise_count / len(labels)

        why = (
            f"Kombinasi parameter {best_params} dipilih terutama berdasarkan DBCV tertinggi "
            f"({metric_value:.4f}), dengan aturan praktis notebook untuk menghindari noise berlebihan."
        )

        meaning = (
            f"HDBSCAN menemukan {n_clusters} cluster secara otomatis dan menandai "
            f"{noise_count} produk ({noise_pct:.1f}%) sebagai noise. Noise bukan otomatis data salah; "
            "produk tersebut hanya tidak cukup mirip/padat untuk dimasukkan ke cluster utama."
        )

        if metric_value > 0.25:
            quality = "DBCV positif menunjukkan struktur density cluster cukup mendukung."
        elif metric_value >= 0:
            quality = "DBCV masih positif tetapi dekat nol, sehingga pemisahan density cenderung lemah."
        else:
            quality = "DBCV negatif menunjukkan struktur density cluster kurang kuat."

        extra = (
            "Produk noise sebaiknya diperiksa sebagai pola penjualan yang tidak umum, "
            "bukan langsung dihapus."
        )

    elif selected_method in {
        "K-Means + DTW",
        "K-Medoids + DTW",
    }:
        k = best_params.get("k", n_clusters)

        why = (
            f"Jumlah cluster k={k} dipilih karena memberikan Silhouette DTW terbaik "
            f"({metric_value:.4f}) dibanding kandidat k lain."
        )

        if selected_method == "K-Medoids + DTW":
            meaning = (
                f"Produk dibagi menjadi {n_clusters} kelompok pola penjualan. "
                "Setiap cluster memiliki medoid, yaitu produk aktual yang paling mewakili cluster."
            )
        else:
            meaning = (
                f"Produk dibagi menjadi {n_clusters} kelompok berdasarkan kemiripan pola time series "
                "dengan DTW. DTW memungkinkan pola yang mirip tetapi bergeser waktu tetap dianggap dekat."
            )

        quality = (
            f"Secara praktis, nilai Silhouette {metric_value:.4f} menunjukkan "
            f"{silhouette_reading(metric_value)}."
        )

        extra = (
            "Makna tiap cluster tidak ditentukan oleh nomor cluster, tetapi oleh profil "
            "volume, aktivitas, dan bentuk penjualannya."
        )

    elif selected_method == "k-Shape":
        k = best_params.get("k", n_clusters)

        why = (
            f"k={k} dipilih karena menghasilkan Silhouette SBD tertinggi "
            f"({metric_value:.4f}) pada kandidat yang valid."
        )

        meaning = (
            f"k-Shape menghasilkan {n_clusters} kelompok berdasarkan kemiripan bentuk pola, "
            "bukan besarnya penjualan. Karena itu produk dengan skala penjualan berbeda dapat "
            "masuk cluster yang sama bila bentuk naik-turunnya mirip."
        )

        quality = (
            f"Silhouette SBD {metric_value:.4f} menunjukkan "
            f"{silhouette_reading(metric_value)} pada ruang shape-based distance."
        )

        constant_count = int(np.sum(labels < 0))
        extra = (
            f"Ada {constant_count} profil berlabel -1/khusus bila pola konstan dipisahkan dari fitting. "
            "Label tersebut bukan noise bawaan algoritma k-Shape."
        )

    else:
        why = (
            f"Parameter {best_params} dipilih berdasarkan {metric_name} terbaik "
            "di antara kandidat yang dievaluasi."
        )
        meaning = (
            f"Metode menghasilkan {n_clusters} cluster."
        )
        quality = (
            f"{metric_name} = {metric_value:.4f}."
        )

    return {
        "why": why,
        "meaning": meaning,
        "quality": quality,
        "extra": extra,
    }


def conclusion_text(selected_method, result, profile):
    labels = np.asarray(result["labels"])
    valid = labels[labels >= 0]
    n_clusters = len(np.unique(valid))

    if profile.empty:
        return (
            f"{selected_method} menghasilkan {n_clusters} cluster, tetapi profil cluster "
            "belum dapat diringkas dari data yang tersedia."
        )

    high_row = profile.loc[
        profile["Rata2_Total_Sales"].idxmax()
    ]
    low_row = profile.loc[
        profile["Rata2_Total_Sales"].idxmin()
    ]

    largest_row = profile.loc[
        profile["Jumlah_Produk"].idxmax()
    ]

    conclusion = (
        f"Secara keseluruhan, {selected_method} membentuk {n_clusters} kelompok produk. "
        f"Cluster {int(high_row['Cluster'])} memiliki rata-rata total penjualan tertinggi "
        f"({high_row['Rata2_Total_Sales']:,.1f}), sedangkan Cluster {int(low_row['Cluster'])} "
        f"memiliki rata-rata terendah ({low_row['Rata2_Total_Sales']:,.1f}). "
        f"Cluster terbesar adalah Cluster {int(largest_row['Cluster'])} dengan "
        f"{int(largest_row['Jumlah_Produk'])} produk "
        f"({largest_row['Proporsi_Produk_%']:.1f}% dari produk yang tercluster). "
        "Dengan demikian, nomor cluster sebaiknya tidak dibaca sebagai urutan baik-buruk; "
        "interpretasinya didasarkan pada karakter pola dan volume masing-masing cluster."
    )

    return conclusion


def practical_actions(profile):
    if profile.empty:
        return []

    actions = []

    high = profile.loc[
        profile["Rata2_Total_Sales"].idxmax()
    ]
    low = profile.loc[
        profile["Rata2_Total_Sales"].idxmin()
    ]
    sporadic = profile.loc[
        profile["Rata2_Minggu_Nol"].idxmax()
    ]

    actions.append(
        f"Cluster {int(high['Cluster'])}: volume penjualan relatif tertinggi. "
        "Dapat diprioritaskan untuk ketersediaan stok dan pemantauan permintaan."
    )

    if int(low["Cluster"]) != int(high["Cluster"]):
        actions.append(
            f"Cluster {int(low['Cluster'])}: volume relatif terendah. "
            "Perlu ditinjau apakah produknya memang niche, baru, musiman, atau kurang aktif."
        )

    if int(sporadic["Cluster"]) not in {
        int(high["Cluster"]),
        int(low["Cluster"]),
    }:
        actions.append(
            f"Cluster {int(sporadic['Cluster'])}: rata-rata minggu tanpa penjualan paling tinggi. "
            "Cocok diperiksa sebagai kelompok sporadis/musiman."
        )

    return actions



# ============================================================
# SIDEBAR — ONLY DATA UPLOAD
# ============================================================

st.sidebar.markdown("## Sales Clustering")
st.sidebar.caption(
    "1. Upload data\n\n2. Klik metode di halaman utama\n\n3. Hasil langsung muncul"
)

uploaded = st.sidebar.file_uploader(
    "Upload Sales transaction.csv",
    type=["csv"],
    accept_multiple_files=False,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Tidak perlu upload file hasil clustering."
)


# ============================================================
# HEADER
# ============================================================

hero(
    "Sales Clustering Dashboard",
    "Upload data sekali, lalu klik salah satu metode. "
    "Dashboard langsung memanggil file metode yang terhubung dan menampilkan hasilnya.",
)

if uploaded is None:
    st.info(
        "Upload `Sales transaction.csv` di sidebar terlebih dahulu."
    )
    st.stop()

raw_bytes = uploaded.getvalue()
signature = file_hash(raw_bytes)

df = pd.read_csv(
    io.BytesIO(raw_bytes)
)

try:
    validate_input(df)
except Exception as exc:
    st.error(str(exc))
    st.stop()

if "results" not in st.session_state:
    st.session_state["results"] = {}

if "active_method" not in st.session_state:
    st.session_state["active_method"] = None


# ============================================================
# OVERVIEW
# ============================================================

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        metric_card(
            "Jumlah produk",
            f"{df['Product_Code'].nunique():,}",
            "Unit analisis",
        ),
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        metric_card(
            "Periode",
            "52 minggu",
            "W0 sampai W51",
        ),
        unsafe_allow_html=True,
    )

with c3:
    done_count = len(
        [
            key
            for key in st.session_state["results"]
            if key.startswith(signature)
        ]
    )
    st.markdown(
        metric_card(
            "Metode sudah dijalankan",
            f"{done_count}/6",
            "Pada data yang sedang diupload",
        ),
        unsafe_allow_html=True,
    )

st.write("")

st.markdown(
    """
    <div class="info-box">
        <b>Pilih metode yang ingin dijalankan.</b>
        Cukup klik satu tombol di bawah. Tidak ada langkah kedua.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DIRECT CLICK METHOD BUTTONS
# ============================================================

method_names = list(METHOD_RUNNERS.keys())

for start in range(0, 6, 3):
    cols = st.columns(3)

    for col, method_name in zip(
        cols,
        method_names[start:start+3],
    ):
        short, button_label = METHOD_INFO[method_name]
        key = result_key(signature, method_name)
        already_done = key in st.session_state["results"]

        with col:
            st.markdown(
                f"""
                <div class="method-title">{button_label}</div>
                <div class="method-sub">{short}</div>
                """,
                unsafe_allow_html=True,
            )

            label = (
                f"✓ Buka {button_label}"
                if already_done
                else f"▶ Jalankan {button_label}"
            )

            if st.button(
                label,
                key=f"run_{method_name}",
                use_container_width=True,
            ):
                st.session_state["active_method"] = method_name

                if not already_done:
                    runner = METHOD_RUNNERS[method_name]

                    try:
                        with st.spinner(
                            f"Menjalankan {method_name}..."
                        ):
                            result = runner(df.copy())

                        st.session_state["results"][key] = result

                    except Exception as exc:
                        st.exception(exc)
                        st.stop()

                st.rerun()


# ============================================================
# ACTIVE RESULT
# ============================================================

selected_method = st.session_state["active_method"]

if selected_method is None:
    st.stop()

key = result_key(
    signature,
    selected_method,
)

result = st.session_state["results"].get(key)

if result is None:
    st.stop()

runner = METHOD_RUNNERS[selected_method]
runner_source = inspect.getsourcefile(runner)

st.markdown("---")

st.markdown(
    f"""
    <div class="connected-box">
        <b>{selected_method}</b> sedang ditampilkan.
        Dashboard memanggil langsung file
        <code>{runner_source}</code>.
        Algoritma tidak ditulis ulang di app.py.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# RESULT CARDS
# ============================================================

labels = np.asarray(result["labels"])
valid = labels[labels >= 0]
n_clusters = len(np.unique(valid))
n_special = int(np.sum(labels < 0))

cards = st.columns(4)

best_param_text = ", ".join(
    f"{k}={v}"
    for k, v in result["best_params"].items()
)

values = [
    (
        "Jumlah cluster",
        n_clusters,
        "Tidak termasuk label -1",
    ),
    (
        result["primary_metric_name"],
        f"{result['primary_metric_value']:.4f}",
        "Metrik utama",
    ),
    (
        "Parameter terbaik",
        best_param_text,
        "Hasil tuning metode",
    ),
    (
        "Label -1 / khusus",
        n_special,
        "Noise / profil khusus bila ada",
    ),
]

for col, card in zip(cards, values):
    with col:
        st.markdown(
            metric_card(*card),
            unsafe_allow_html=True,
        )

st.write("")

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📌 Makna & Kesimpulan",
        "📊 Hasil Cluster",
        "🧪 Evaluasi",
        "📈 Pola Penjualan",
        "🔎 Produk",
        "⚖️ Bandingkan",
    ]
)



# ============================================================
# TAB 0 — MAKNA & KESIMPULAN
# ============================================================

with tab0:
    interpretation = model_interpretation(
        selected_method,
        result,
    )

    profile_df = cluster_profile_table(
        result
    )

    st.subheader(
        "Apa arti hasil ini?"
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">Mengapa model ini yang dipilih?</div>
                <div class="insight-text">{interpretation["why"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">Artinya secara sederhana</div>
                <div class="insight-text">{interpretation["meaning"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">Bagaimana membaca kualitas hasil?</div>
                <div class="insight-text">{interpretation["quality"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_d:
        extra_text = interpretation["extra"] or (
            "Tidak ada catatan khusus tambahan untuk metode ini."
        )

        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">Hal yang perlu diperhatikan</div>
                <div class="insight-text">{extra_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader(
        "Profil setiap cluster"
    )

    st.caption(
        "Label profil di bawah bersifat relatif terhadap cluster lain pada data yang sama. "
        "Profil dibuat dari penjualan asli W0–W51, bukan mengubah hasil clustering."
    )

    if not profile_df.empty:
        profile_show = profile_df[
            [
                "Cluster",
                "Jumlah_Produk",
                "Proporsi_Produk_%",
                "Rata2_Total_Sales",
                "Rata2_Penjualan_Mingguan",
                "Rata2_Minggu_Nol",
                "Profil",
            ]
        ].copy()

        profile_show = profile_show.rename(
            columns={
                "Proporsi_Produk_%": "Proporsi (%)",
                "Rata2_Total_Sales": "Rata-rata Total Sales",
                "Rata2_Penjualan_Mingguan": "Rata-rata Sales/Minggu",
                "Rata2_Minggu_Nol": "Rata-rata Minggu Nol",
            }
        )

        st.dataframe(
            profile_show.round(2),
            use_container_width=True,
            hide_index=True,
        )

        fig_profile = px.scatter(
            profile_df,
            x="Rata2_Minggu_Nol",
            y="Rata2_Total_Sales",
            size="Jumlah_Produk",
            text=profile_df["Cluster"].apply(
                lambda x: f"C{int(x)}"
            ),
            hover_data=[
                "Profil",
                "Proporsi_Produk_%",
                "Rata2_CV",
            ],
        )

        fig_profile.update_layout(
            title=(
                "Peta karakter cluster: volume penjualan vs minggu tanpa penjualan"
            ),
            xaxis_title="Rata-rata jumlah minggu tanpa penjualan",
            yaxis_title="Rata-rata total penjualan",
        )

        st.plotly_chart(
            plot_layout(
                fig_profile,
                430,
            ),
            use_container_width=True,
        )

    st.subheader(
        "Kesimpulan"
    )

    st.markdown(
        f"""
        <div class="conclusion-box">
            <b>Kesimpulan otomatis dari hasil running:</b><br>
            {conclusion_text(selected_method, result, profile_df)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    actions = practical_actions(
        profile_df
    )

    if actions:
        action_html = "<br>".join(
            f"• {a}"
            for a in actions
        )

        st.markdown(
            f"""
            <div class="action-box">
                <b>Implikasi praktis yang bisa dibaca dari cluster:</b><br>
                {action_html}
                <br><br>
                <span style="font-size:.82rem;">
                Catatan: ini adalah interpretasi deskriptif dari pola penjualan, bukan keputusan bisnis otomatis.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )



# ============================================================
# TAB 1 — CLUSTER RESULT
# ============================================================

with tab1:
    counts = (
        result["result_df"][
            result["cluster_col"]
        ]
        .value_counts()
        .sort_index()
        .rename_axis("Cluster")
        .reset_index(name="Jumlah Produk")
    )

    counts["Label"] = counts["Cluster"].apply(
        lambda x: (
            "Noise / khusus (-1)"
            if float(x) < 0
            else f"Cluster {int(x)}"
        )
    )

    fig = px.bar(
        counts,
        x="Label",
        y="Jumlah Produk",
        text="Jumlah Produk",
    )

    fig.update_traces(
        marker_color=[
            "#F4B942"
            if float(x) < 0
            else "#1E5A8C"
            for x in counts["Cluster"]
        ],
        textposition="outside",
    )

    fig.update_layout(
        title=f"Distribusi cluster — {selected_method}"
    )

    st.plotly_chart(
        plot_layout(fig),
        use_container_width=True,
    )

    if not result["summary_df"].empty:
        st.dataframe(
            result["summary_df"].round(3),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# TAB 2 — EVALUATION
# ============================================================

with tab2:
    st.caption(
        "Hasil evaluasi berasal langsung dari file metode yang dijalankan."
    )

    eval_df = result["evaluation_df"]

    st.dataframe(
        eval_df.round(6),
        use_container_width=True,
        hide_index=True,
    )

    x_candidates = [
        c for c in [
            "k",
            "c",
            "n_components",
            "min_cluster_size",
        ]
        if c in eval_df.columns
    ]

    numeric_candidates = [
        c
        for c in eval_df.columns
        if pd.api.types.is_numeric_dtype(
            eval_df[c]
        )
    ]

    if x_candidates and numeric_candidates:
        metric = st.selectbox(
            "Metrik evaluasi",
            numeric_candidates,
            key=f"metric_{selected_method}",
        )

        fig = px.line(
            eval_df,
            x=x_candidates[0],
            y=metric,
            markers=True,
        )

        fig.update_layout(
            title=f"{metric} — {selected_method}"
        )

        st.plotly_chart(
            plot_layout(fig),
            use_container_width=True,
        )


# ============================================================
# TAB 3 — PATTERN
# ============================================================

with tab3:
    pattern_df = result["pattern_df"]

    if pattern_df.empty:
        st.info(
            "Tidak ada pola cluster valid yang dapat ditampilkan."
        )
    else:
        fig = px.line(
            pattern_df,
            x="Minggu",
            y="Rata-rata Penjualan",
            color="Cluster",
        )

        fig.update_layout(
            title=f"Pola rata-rata W0-W51 — {selected_method}"
        )

        st.plotly_chart(
            plot_layout(fig, 470),
            use_container_width=True,
        )

        st.caption(
            "Grafik interpretasi menggunakan penjualan asli W0-W51."
        )


# ============================================================
# TAB 4 — PRODUCT
# ============================================================

with tab4:
    product_codes = sorted(
        result["result_df"][
            "Product_Code"
        ].astype(str).unique()
    )

    product = st.selectbox(
        "Pilih Product_Code",
        product_codes,
        key=f"product_{selected_method}",
    )

    row = result["result_df"][
        result["result_df"][
            "Product_Code"
        ].astype(str)
        == product
    ].iloc[0]

    series = pd.DataFrame({
        "Minggu": np.arange(52),
        "Penjualan": row[
            WEEK_COLS
        ].to_numpy(dtype=float),
    })

    fig = px.area(
        series,
        x="Minggu",
        y="Penjualan",
        markers=True,
    )

    fig.update_traces(
        line=dict(
            color="#1E5A8C",
            width=2.5,
        ),
        fillcolor="rgba(30,90,140,.12)",
    )

    fig.update_layout(
        title=f"Pola produk {product}"
    )

    st.plotly_chart(
        plot_layout(fig, 390),
        use_container_width=True,
    )

    detail_cols = [
        "Product_Code",
        result["cluster_col"],
    ]

    detail_cols += [
        c
        for c in result["result_df"].columns
        if (
            "Membership" in c
            or "Probability" in c
            or "Outlier" in c
            or "Distance" in c
            or "Entropy" in c
            or "Status" in c
        )
    ]

    detail_cols = list(
        dict.fromkeys(
            c
            for c in detail_cols
            if c in result["result_df"].columns
        )
    )

    st.dataframe(
        row[detail_cols].to_frame("Nilai"),
        use_container_width=True,
    )

    current_cluster = row[result["cluster_col"]]

    st.markdown(
        f"""
        <div class="conclusion-box">
            <b>Makna produk {product}:</b><br>
            Produk ini masuk ke <b>Cluster {current_cluster}</b> pada metode <b>{selected_method}</b>.
            Gunakan grafik 52 minggu di atas untuk melihat apakah penjualannya cenderung aktif,
            sporadis, stabil, atau memiliki lonjakan pada minggu tertentu.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# TAB 5 — ARI COMPARISON
# ============================================================

with tab5:
    same_data_results = {
        k.split("::", 1)[1]: v
        for k, v in st.session_state["results"].items()
        if k.startswith(signature)
    }

    if len(same_data_results) < 2:
        st.info(
            "Klik minimal dua metode agar hasilnya bisa dibandingkan dengan ARI."
        )
    else:
        methods = list(
            same_data_results.keys()
        )

        ari = pd.DataFrame(
            np.nan,
            index=methods,
            columns=methods,
        )

        for m1 in methods:
            for m2 in methods:
                if m1 == m2:
                    ari.loc[m1, m2] = 1.0
                    continue

                r1 = same_data_results[m1]
                r2 = same_data_results[m2]

                a = pd.DataFrame({
                    "Product_Code":
                        r1["result_df"]["Product_Code"].astype(str),
                    "label1":
                        pd.to_numeric(
                            r1["result_df"][
                                r1["cluster_col"]
                            ],
                            errors="coerce",
                        ),
                })

                b = pd.DataFrame({
                    "Product_Code":
                        r2["result_df"]["Product_Code"].astype(str),
                    "label2":
                        pd.to_numeric(
                            r2["result_df"][
                                r2["cluster_col"]
                            ],
                            errors="coerce",
                        ),
                })

                joined = a.merge(
                    b,
                    on="Product_Code",
                    how="inner",
                )

                joined = joined[
                    joined["label1"].notna()
                    & joined["label2"].notna()
                    & (joined["label1"] >= 0)
                    & (joined["label2"] >= 0)
                ]

                if (
                    len(joined) >= 2
                    and joined["label1"].nunique() >= 2
                    and joined["label2"].nunique() >= 2
                ):
                    ari.loc[m1, m2] = adjusted_rand_score(
                        joined["label1"],
                        joined["label2"],
                    )

        fig = go.Figure(
            data=go.Heatmap(
                z=ari.to_numpy(dtype=float),
                x=ari.columns,
                y=ari.index,
                zmin=-1,
                zmax=1,
                text=np.round(
                    ari.to_numpy(dtype=float),
                    3,
                ),
                texttemplate="%{text}",
                colorbar=dict(title="ARI"),
            )
        )

        fig.update_layout(
            title="Agreement antar-metode yang sudah diklik"
        )

        st.plotly_chart(
            plot_layout(fig, 560),
            use_container_width=True,
        )


# ============================================================
# DOWNLOAD + PROOF
# ============================================================

st.markdown("---")

st.download_button(
    "Download hasil metode ini",
    data=result["result_df"].to_csv(index=False).encode("utf-8"),
    file_name=(
        selected_method
        .lower()
        .replace(" ", "_")
        .replace("+", "plus")
        + "_hasil.csv"
    ),
    mime="text/csv",
)

with st.expander("Bukti koneksi metode"):
    st.write("Fungsi yang dipanggil:")
    st.code(
        f"{runner.__module__}.{runner.__name__}",
        language="python",
    )

    st.write("File sumber metode:")
    st.code(
        str(runner_source),
        language=None,
    )

    st.write("Catatan:")
    st.write(
        result.get("notes", "—")
    )
