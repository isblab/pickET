import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

DEFAULT_NTHREADS = 8


def set_larger_cluster_as_bg(segmentation: np.ndarray) -> np.ndarray:
    lids, lid_counts = np.unique(segmentation.reshape(-1), return_counts=True)
    largest_cluster_id, largest_cluster_count = -1, -1

    for lid, count in zip(lids, lid_counts):
        if count > largest_cluster_count:
            largest_cluster_id = lid
            largest_cluster_count = count

    if largest_cluster_id == 1:
        segmentation = np.where(segmentation == 1, 0, 1)
    return segmentation


def fit_kmeans_clustering(features: np.ndarray) -> KMeans:
    with threadpool_limits(limits=DEFAULT_NTHREADS):
        clusterer = KMeans(n_clusters=2, random_state=42)
        clusterer.fit(features.astype(np.float64))
    return clusterer


def fit_gmm_clustering(features: np.ndarray) -> GaussianMixture:
    with threadpool_limits(limits=DEFAULT_NTHREADS):
        clusterer = GaussianMixture(n_components=2, random_state=42)
        clusterer.fit(features.astype(np.float64))
    return clusterer
