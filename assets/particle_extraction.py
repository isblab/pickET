import ndjson
import numpy as np
import scipy.ndimage as nd
import skimage.feature as ft
from sklearn.cluster import MeanShift
from skimage.segmentation import watershed


def do_connected_component_labeling(segmentation: np.ndarray) -> tuple[np.ndarray, int]:
    labeled_array, num_features = nd.label(segmentation)  # type:ignore
    return labeled_array, num_features


def do_watershed_segmentation(
    segmentation: np.ndarray, min_distance: int
) -> tuple[np.ndarray, int]:
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
    elif pex_params["mode"] == "watershed":
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
    return np.array(centroids)


def write_coords_as_ndjson(coords: np.ndarray, out_fname: str) -> None:
    lines = []
    for coord in coords:
        lines.append(
            {
                "type": "orientedPoint",
                "location": {"x": coord[2], "y": coord[1], "z": coord[0]},
            }
        )

    with open(out_fname, "w") as out_annot_f:
        ndjson.dump(lines, out_annot_f)
