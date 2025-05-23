import os
import sys
import yaml
import ndjson
import numpy as np
from rich.progress import track
from scipy.spatial.distance import cdist

from assets import utils
import assessment_utils


def read_annotations_w_deets(fname: str) -> dict[str, np.ndarray]:
    with open(fname, "r") as in_annot_f:
        annotations = ndjson.load(in_annot_f)

    p_deets = {}
    for ln in annotations:
        if "particle_id" in ln:
            particle_id = ln.get("particle_id")
        elif "particle_name" in ln:
            particle_id = ln.get("particle_name")
        else:
            raise KeyError("Particle annotations dont have a label on them")

        if particle_id not in p_deets:
            p_deets[particle_id] = [
                np.array(
                    [ln["location"]["z"], ln["location"]["y"], ln["location"]["x"]]
                )
            ]
        else:
            p_deets[particle_id].append(
                np.array(
                    [ln["location"]["z"], ln["location"]["y"], ln["location"]["x"]]
                )
            )

    return p_deets


def convert_centroids_dict_to_arr(centroids: list[dict]) -> np.ndarray:
    out_arr = np.nan * np.ones((len(centroids), 3))
    for idx, c in enumerate(centroids):
        out_arr[idx] = np.array([c["z"], c["y"], c["x"]])
    return out_arr


def main():
    ### Input block
    params_fname = sys.argv[1]
    dataset_name = sys.argv[2]
    params = utils.load_params_from_yaml(params_fname)

    angstrom_threshold = params["threshold_in_angstrom"]
    parent_path = params["parent_path"]

    out_dir = f"/home/shreyas/Projects/mining_tomograms/pickET/partice_wise_recall/{dataset_name}/raw/"
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    ### Processing block
    groups = assessment_utils.separate_files_into_groups(parent_path)
    for group_name, pc_fname in groups.items():
        results = {}
        feature_extraction_method = ""
        clustering_method = ""
        particle_extraction_method = ""
        ### Processing block
        for idx, target in enumerate(
            track(pc_fname, description=f"Processing {group_name}")
        ):
            pred_centroids_dicts, pred_metadata = assessment_utils.load_predictions(
                target
            )
            pred_centroids = convert_centroids_dict_to_arr(pred_centroids_dicts)
            gt_fpath = assessment_utils.get_ground_truth_fpath(
                pred_metadata["tomogram_path"],
                annot_dir_head=params["annot_dir_head"],
            )
            true_particles = read_annotations_w_deets(gt_fpath)
            threshold = assessment_utils.get_voxel_threshold(
                angstrom_threshold, pred_metadata["voxel_size"]
            )

            feature_extraction_method = pred_metadata["fex_mode"]
            clustering_method = pred_metadata["clustering_method"]
            particle_extraction_method = pred_metadata["pex_mode"]
            zslice_lb = pred_metadata.get("lower_z-slice_limit")
            zslice_ub = pred_metadata.get("upper_z-slice_limit")
            if zslice_lb == None:
                zslice_lb = "None"
            if zslice_ub == None:
                zslice_ub = "None"

            tomo_shape = tuple(np.array(pred_metadata["tomogram_shape"]).tolist())
            random_centroids = assessment_utils.get_random_centroids(
                tomo_shape, num_centroids=len(pred_centroids)
            )
            for particle_id, true_centroids in true_particles.items():
                true_centroids = np.array(true_centroids)
                true_centroids = assessment_utils.zslice_filter_ground_truth_centroids(
                    true_centroids, zslice_lb, zslice_ub
                )
                distances = cdist(pred_centroids, true_centroids, metric="euclidean")
                _, recall, _ = assessment_utils.compute_precision_recall_f1score(
                    distances,
                    threshold,
                    num_pred=len(pred_centroids),
                    num_true=len(true_centroids),
                )
                random_distances = cdist(
                    random_centroids, true_centroids, metric="euclidean"
                )
                _, random_recall, _ = assessment_utils.compute_precision_recall_f1score(
                    random_distances,
                    threshold,
                    num_pred=len(random_centroids),
                    num_true=len(true_centroids),
                )

                if particle_id not in results:
                    results[particle_id] = {
                        "Recalls": [float(recall)],
                        "Random recalls": [float(random_recall)],
                        "Average recall": 0.0,
                        "Average recall ratio": 0.0,
                        "Standard deviation on recall": 0.0,
                    }
                else:
                    results[particle_id]["Recalls"].append(recall)
                    results[particle_id]["Random recalls"].append(random_recall)

        for particle_id in results:
            results[particle_id]["Average recall"] = float(
                np.mean(np.array(results[particle_id]["Recalls"]))
            )
            results[particle_id]["Standard deviation on recall"] = float(
                np.std(np.array(results[particle_id]["Recalls"]))
            )

            r1 = np.array(results[particle_id]["Recalls"])
            r2 = np.array(results[particle_id]["Random recalls"]) + 1e-8
            results[particle_id]["Average recall ratio"] = float(np.mean(r1 / r2))

        with open(
            os.path.join(
                out_dir,
                f"particlewise_recall_{feature_extraction_method}_{clustering_method}_{particle_extraction_method}.yaml",
            ),
            "w",
        ) as outf:
            yaml.dump(results, outf)


if __name__ == "__main__":
    main()
