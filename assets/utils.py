import yaml
import ndjson
import psutil
import numpy as np


def load_params_from_yaml(param_file_path: str) -> dict:
    with open(param_file_path, "r") as paramf:
        return yaml.safe_load(paramf)


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


def check_memory_availability(
    num_windows: int, num_features: int
) -> tuple[bool, float, float]:
    is_available = False
    memory_reqd = (
        num_windows * num_features * 8
        + 2 * num_features * 8
        + 2 * (num_features**2) * 8
    )

    memory_available = psutil.virtual_memory().available
    if memory_reqd < memory_available:
        is_available = True

    return is_available, memory_available / (1024**3), memory_reqd / (1024**3)
