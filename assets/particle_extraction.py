import numpy as np
import scipy.ndimage as nd
import skimage.feature as ft
from sklearn.cluster import MeanShift
from skimage.segmentation import watershed


def do_connected_component_labeling(segmentation: np.ndarray) -> tuple[np.ndarray, int]:
    print("Performing connected component labeling")
    labeled_array, num_features = nd.label(segmentation)  # type:ignore
    return labeled_array, num_features


def do_watershed_segmentation(
    segmentation: np.ndarray, min_distance: int
) -> tuple[np.ndarray, int]:
    print("Performing watershed segmentation")
    edt = np.array(nd.distance_transform_edt(segmentation))
    local_maxima = ft.peak_local_max(edt, min_distance=min_distance)
    mask = np.zeros(edt.shape, dtype=bool)
    mask[tuple(local_maxima.T)] = True
    markers, _ = nd.label(mask)  # type:ignore

    watershed_seg_mask = watershed(-edt, markers, mask=segmentation)
    return watershed_seg_mask, len(np.unique(watershed_seg_mask))


def do_mean_shift_clustering(
    segmentation: np.ndarray, bandwidth: float
) -> tuple[np.ndarray, int]:
    print("Performing mean shift clustering")
    particle_voxel_idx = np.array(np.where(segmentation == 1)).T
    clusterer = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    clusterer.fit(particle_voxel_idx)
    labels, _ = clusterer.labels_, clusterer.cluster_centers_.shape[0]
    out_seg = np.zeros(segmentation.shape)
    out_seg[
        particle_voxel_idx[:, 0], particle_voxel_idx[:, 1], particle_voxel_idx[:, 2]
    ] = labels
    return out_seg, len(np.unique(out_seg))


def do_instance_segmentation(segmentation, pex_params) -> tuple[np.ndarray, int]:
    if pex_params["mode"] == "connected_component_labeling":
        instance_seg, num_objects = do_connected_component_labeling(segmentation)
    elif pex_params["mode"] == "watershed_segmentation":
        instance_seg, num_objects = do_watershed_segmentation(
            segmentation, pex_params["min_distance"]
        )
    elif pex_params["mode"] == "mean_shift_clustering":
        instance_seg, num_objects = do_mean_shift_clustering(
            segmentation, pex_params["bandwidth"]
        )
    else:
        raise ValueError(
            f"Particle extraction mode {pex_params['mode']} not supported.\
                                 Choose from 'connected_component_labeling', 'watershed', 'mean_shift_clustering'"
        )
    return instance_seg, num_objects


def get_centroids(instance_segmentation: np.ndarray, num_objects: int) -> np.ndarray:
    centroids = nd.center_of_mass(
        instance_segmentation, instance_segmentation, range(1, num_objects + 1)
    )
    centroids = np.array(centroids)
    if np.any(np.isnan(centroids)):
        centroids = centroids[np.invert(np.any(np.isnan(centroids), axis=1))]
    return centroids


def extract_subtomograms(
    centroids: list[dict], subtomogram_size: int, tomogram: np.ndarray
) -> list[np.ndarray]:
    half_size = subtomogram_size // 2
    subtomograms = []
    for centroid in centroids:
        x, y, z = int(centroid["x"]), int(centroid["y"]), int(centroid["z"])
        x_min, x_max = (x - half_size), (x + half_size + 1)
        y_min, y_max = (y - half_size), (y + half_size + 1)
        z_min, z_max = (z - half_size), (z + half_size + 1)

        st = tomogram[z_min:z_max, y_min:y_max, x_min:x_max]
        if st.shape == (subtomogram_size, subtomogram_size, subtomogram_size):
            subtomograms.append(st)

    return subtomograms
