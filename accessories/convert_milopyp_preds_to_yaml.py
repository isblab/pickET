import os
import sys
import yaml
import numpy as np
import pandas as pd
from rich.progress import track
from picket.core import utils


def _read_milopyp_inputs_file(fpath: str):
    contents = pd.read_csv(fpath, sep="\t")
    contents.reset_index()
    out_dict = {v["image_name"]: v["rec_path"] for _, v in contents.iterrows()}
    return out_dict


def main():
    dataset_id = str(sys.argv[1])
    parent_path = sys.argv[2]
    time_log_fname = sys.argv[3]

    milopyp_inputs_fpath = os.path.join(parent_path, "data", f"input_{dataset_id}.txt")
    pred_npz_fpath = os.path.join(
        parent_path, "exp", "simsiam3d", dataset_id, "all_output_info.npz"
    )

    output_dir = os.path.join(
        parent_path, "exp", "simsiam3d", dataset_id, "predicted_particles"
    )
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    tomo_paths = _read_milopyp_inputs_file(milopyp_inputs_fpath)
    with open(time_log_fname, "r") as tlf:
        time_logs = yaml.safe_load(tlf)

    preds = np.load(pred_npz_fpath)
    all_coords = preds["coords"]
    all_coords_tomo_keys = preds["name"]

    for k, tomo_path in track(tomo_paths.items()):
        tomo, voxel_sizes = utils.load_tomogram(tomo_path)
        out_dict = {
            "metadata": {
                "source": "MiLoPYP",
                "tomogram_path": tomo_path,
                "tomogram_shape": [int(nvox) for nvox in tomo.shape],
                "voxel_size": [float(vs) for vs in voxel_sizes],
                "time_taken_for_s1": time_logs[dataset_id]["s1"],
                "time_taken_for_s2": time_logs[dataset_id]["s2"],
            },
            "Predicted_Particle_Centroid_Coordinates": [],
        }
        for idx in range(len(all_coords_tomo_keys)):
            if all_coords_tomo_keys[idx] == k:
                coord = all_coords[idx]
                x, z, y = coord[0], coord[1], coord[2]
                out_dict["Predicted_Particle_Centroid_Coordinates"].append(
                    {"x": int(x), "y": int(y), "z": int(z)}
                )

        with open(
            os.path.join(output_dir, f"{k}_milopyp_pred_coords.yaml"), "w"
        ) as outf:
            yaml.dump(out_dict, outf)


if __name__ == "__main__":
    main()
