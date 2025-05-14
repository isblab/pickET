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

    coords = np.nan * np.ones((len(annotations), 3), dtype=np.int32)
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
) -> dict:
    out_dict = {}
    out_dict["metadata"] = {}
    for k, v in metadata.items():
        if isinstance(v, np.generic):
            out_dict["metadata"][k] = v.item()
        else:
            out_dict["metadata"][k] = v

    out_dict["Predicted_Particle_Centroid_Coordinates"] = []
    for coord in coords:
        out_dict["Predicted_Particle_Centroid_Coordinates"].append(
            {"x": int(coord[2]), "y": int(coord[1]), "z": int(coord[0])}
        )

    return out_dict


def get_neighborhoods(
    tomo: np.ndarray,
    neighborhood_size: int,
    max_num_neighborhoods_for_fitting: Optional[int] = None,
) -> tuple:
    if neighborhood_size % 2 == 0:
        raise ValueError(
            f"Please set neighborhood_size to an odd integer. It was set to {neighborhood_size}"
        )
    neighborhoods = view_as_windows(tomo, neighborhood_size)
    preshape = neighborhoods.shape[:3]
    neighborhoods = neighborhoods.reshape(
        -1, neighborhood_size, neighborhood_size, neighborhood_size
    )

    if max_num_neighborhoods_for_fitting is not None:
        if max_num_neighborhoods_for_fitting >= len(neighborhoods):
            max_num_neighborhoods_for_fitting = None
        else:
            neighborhoods = subsample_neighborhoods(
                neighborhoods, max_num_neighborhoods_for_fitting
            )

    return neighborhoods, preshape, max_num_neighborhoods_for_fitting


def subsample_neighborhoods(
    neighborhoods: np.ndarray, num_output_neighborhoods: int
) -> np.ndarray:
    idxs = np.random.choice(len(neighborhoods), num_output_neighborhoods, replace=False)
    return neighborhoods[idxs]


def read_yaml_coords(pred_coords_fname: str) -> np.ndarray:
    with open(pred_coords_fname, "r") as pred_coords_f:
        annotations = yaml.safe_load(pred_coords_f)[
            "Predicted_Particle_Centroid_Coordinates"
        ]

    coords = np.nan * np.ones((len(annotations), 3), dtype=np.int32)
    for idx, ln in enumerate(annotations):
        coords[idx] = np.array([ln["z"], ln["y"], ln["x"]])
    if np.any(np.isnan(coords)):
        raise ValueError("Something went wrong when reading coords")

    return coords
