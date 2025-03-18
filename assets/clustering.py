import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def _set_larger_cluster_as_bg(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels)
    if counts[1] > counts[0]:
        labels = np.where(labels == 0, 1, 0)
    return labels


def do_kmeans_clustering(features: np.ndarray) -> np.ndarray:
    print("\tInitiating KMeans clustering")
    clusterer = KMeans(n_clusters=2)
    labels = clusterer.fit_predict(features)
    labels = _set_larger_cluster_as_bg(labels)
    return labels


def do_gmm_clustering(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    print("\tInitiating GMM clustering")
    clusterer = GaussianMixture(n_components=2)
    labels = clusterer.fit_predict(features)
    probabilities = clusterer.predict_proba(features)
    labels = _set_larger_cluster_as_bg(labels)

    return labels, probabilities
