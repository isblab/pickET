import os
import sys
import yaml
import numpy as np
import pandas as pd
from rich.progress import track
from assets import utils


def _read_milopyp_inputs_file(fpath: str):
    contents = pd.read_csv(fpath, sep="\t")
    contents.reset_index()
    out_dict = {v["image_name"]: v["rec_path"] for _, v in contents.iterrows()}
    return out_dict


def main():
    pred_npz_fpath = sys.argv[1]
    milopyp_inputs_fpath = sys.argv[2]
    output_dir = sys.argv[3]
    output_dir = os.path.join(output_dir, "pred_centroids")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    tomo_paths = _read_milopyp_inputs_file(milopyp_inputs_fpath)

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
