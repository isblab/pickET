import numpy as np
from picket.core import clustering


def test_set_larger_cluster_as_bg():
    segmentation1 = np.zeros((10, 10, 10), dtype=np.int16)
    segmentation2 = np.zeros((10, 10, 10), dtype=np.int16)
    coords_less = np.random.randint(0, 10, (100, 3))
    coords_more = np.random.randint(0, 10, (750, 3))

    segmentation1[coords_less[:, 0], coords_less[:, 1], coords_less[:, 2]] = 1
    segmentation2[coords_more[:, 0], coords_more[:, 1], coords_more[:, 2]] = 1

    segmentation3 = np.pad(segmentation1, 1, mode="constant", constant_values=-1)
    segmentation4 = np.pad(segmentation2, 1, mode="constant", constant_values=-1)

    flipped_segmentation1 = clustering.set_larger_cluster_as_bg(segmentation1)
    flipped_segmentation2 = clustering.set_larger_cluster_as_bg(segmentation2)

    flipped_segmentation3 = clustering.set_larger_cluster_as_bg(segmentation3)
    flipped_segmentation4 = clustering.set_larger_cluster_as_bg(segmentation4)

    assert isinstance(flipped_segmentation1, np.ndarray)
    assert isinstance(flipped_segmentation2, np.ndarray)
    assert np.array_equal(segmentation1, flipped_segmentation1)
    assert np.array_equal(segmentation2, int(1) - flipped_segmentation2)
    assert (flipped_segmentation3[0] == -1).all()
    assert (flipped_segmentation4[0] == -1).all()


def test_fit_kmeans_clustering():
    n_cl1, n_cl2 = 50, 100
    features1 = np.random.uniform(1, 3, (n_cl1, 3))
    features2 = np.random.uniform(7, 9, (n_cl2, 3))
    features = np.concatenate((features1, features2), axis=0)

    kmeans = clustering.fit_kmeans_clustering(features)
    kmeans_pred = kmeans.predict(features)
    assert list(sorted(np.bincount(kmeans_pred))) == [n_cl1, n_cl2]


def test_fit_gmm_clustering():
    n_cl1, n_cl2 = 50, 100
    features1 = np.random.uniform(1, 3, (n_cl1, 3))
    features2 = np.random.uniform(7, 9, (n_cl2, 3))
    features = np.concatenate((features1, features2), axis=0)

    gmm = clustering.fit_gmm_clustering(features)
    gmm_pred = gmm.predict(features)
    assert list(sorted(np.bincount(gmm_pred))) == [n_cl1, n_cl2]
