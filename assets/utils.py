import yaml
import h5py
import ndjson
import mrcfile
import numpy as np
from skimage.util import view_as_windows
from typing import Optional


def load_params_from_yaml(param_file_path: str) -> dict:
    with open(param_file_path, "r") as paramf:
        return yaml.safe_load(paramf)


def load_tomogram(tomogram_path: str) -> tuple[np.ndarray, np.ndarray]:
    with mrcfile.open(tomogram_path, "r", permissive=True) as mrcf:
        tomogram = np.array(mrcf.data)
        vxs = mrcf.voxel_size
        voxel_sizes = np.zeros(3, dtype=np.float32)
        for i, ax in enumerate(("z", "y", "x")):
            voxel_sizes[i] = vxs[ax]
    return tomogram, voxel_sizes


def read_ndjson_coords(fname: str) -> np.ndarray:
    with open(fname, "r") as in_annot_f:
        annotations = ndjson.load(in_annot_f)

    coords = np.nan * np.ones((len(annotations), 3))
    for idx, ln in enumerate(annotations):
        coords[idx] = np.array(
            [ln["location"]["z"], ln["location"]["y"], ln["location"]["x"]]
        )

    if np.any(np.isnan(coords)):
        raise ValueError("Something went wrong when reading coords")

    return coords


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


def prepare_out_coords(
    coords: np.ndarray,
    metadata: dict,
    z_lb: Optional[int] = None,
    z_ub: Optional[int] = None,
) -> dict:
    out_dict = {}
    out_dict["metadata"] = {}
    for k, v in metadata.items():
        if isinstance(v, np.ndarray):
            out_dict["metadata"][k] = v.tolist()
        elif isinstance(v, np.generic):
            out_dict["metadata"][k] = v.item()
        else:
            out_dict["metadata"][k] = v

    out_dict["metadata"]["z_lb_for_particle_extraction"] = z_lb
    out_dict["metadata"]["z_ub_for_particle_extraction"] = z_ub

    out_dict["PredictedCentroidCoordinates"] = []
    for coord in coords:
        out_dict["PredictedCentroidCoordinates"].append(
            {"x": int(coord[2]), "y": int(coord[1]), "z": int(coord[0])}
        )

    return out_dict


def get_windows(
    tomo: np.ndarray,
    window_size: int,
    max_num_windows_for_fitting: Optional[int] = None,
) -> tuple[np.ndarray, tuple]:
    if window_size % 2 == 0:
        raise ValueError(
            f"Please set window_size to an odd integer. It was set to {window_size}"
        )
    windows = view_as_windows(tomo, window_size)
    preshape = windows.shape[:3]
    windows = windows.reshape(-1, window_size, window_size, window_size)
    if max_num_windows_for_fitting is not None and max_num_windows_for_fitting < len(
        windows
    ):
        windows = subsample_windows(windows, max_num_windows_for_fitting)
    return windows, preshape


def subsample_windows(windows: np.ndarray, num_output_windows: int) -> np.ndarray:
    idxs = np.random.choice(len(windows), num_output_windows, replace=False)
    return windows[idxs]


def save_segmentation(
    output_fname: str,
    segmentation: np.ndarray,
    tomogram_path: str,
    voxel_size: np.ndarray,
    window_size: int,
    feature_extraction_params: dict,
    clustering_method: str,
    time_taken_for_s1: Optional[float] = None,
    time_taken_for_s2: Optional[float] = None,
    max_num_windows_for_fitting: Optional[int] = None,
) -> None:
    metadata = {
        "tomogram_path": tomogram_path,
        "voxel_size": voxel_size,
        "window_size": window_size,
        "feature_extraction_params": feature_extraction_params,
        "clustering_method": clustering_method,
        "max_num_windows_for_fitting": max_num_windows_for_fitting,
        "time_taken_for_s1": time_taken_for_s1,
        "time_taken_for_s2": time_taken_for_s2,
    }
    with h5py.File(output_fname, "w") as f:
        out_dataset = f.create_dataset("segmentation", data=segmentation)

        for k, v in metadata.items():
            if v is None:
                v = -1

            if k == "feature_extraction_params":
                for k1, v1 in v.items():  # type: ignore
                    k1 = f"fexparams_{k1}"
                    out_dataset.attrs[k1] = v1
            else:
                out_dataset.attrs[k] = v


def load_h5file(fname: str) -> tuple[np.ndarray, dict]:
    metadata = {}
    with h5py.File(fname, "r") as h5f:
        dataset = h5f["segmentation"]
        seg = np.array(dataset)
        for k in dataset.attrs:
            metadata[k] = dataset.attrs[k]

    return seg, metadata
