
from .kmeans_dtw import run as run_kmeans_dtw
from .kmedoids_dtw import run as run_kmedoids_dtw
from .hdbscan_dtw import run as run_hdbscan_dtw
from .fuzzy_cmeans import run as run_fuzzy_cmeans
from .gmm import run as run_gmm
from .kshape import run as run_kshape


METHOD_RUNNERS = {
    "K-Means + DTW": run_kmeans_dtw,
    "K-Medoids + DTW": run_kmedoids_dtw,
    "HDBSCAN + DTW": run_hdbscan_dtw,
    "Fuzzy C-Means": run_fuzzy_cmeans,
    "Gaussian Mixture Model": run_gmm,
    "k-Shape": run_kshape,
}
