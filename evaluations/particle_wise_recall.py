import os
import sys
import yaml
import ndjson
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist

from assets import utils
import assessment_utils


def read_annotations_ndjson_file(fname: str) -> dict[str, np.ndarray]:
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


def main():
    ### Input block
    params_fname = sys.argv[1]
    dataset_name = sys.argv[2]
    params = utils.load_params_from_yaml(params_fname)

    angstrom_threshold = params["threshold_in_angstrom"]
    inputs = params["inputs"]
    clustering_method = params["clustering_method"]
    particle_extraction_method = params["particle_extraction_method"]
    feature_extraction_method = params["feature_extraction_method"]

    results = {}
    ### Processing block
    for target in tqdm(inputs):
        # print(f"Processing prediction {idx+1}/{len(inputs)}")
        threshold = assessment_utils.get_voxel_threshold(
            target["tomogram"], angs_threshold=angstrom_threshold
        )
        pred_centroids = utils.read_ndjson_coords(target["predicted_centroids"])
        true_particles = read_annotations_ndjson_file(target["true_centroids"])

        zslice_lb = target.get("lower_z-slice_limit")
        zslice_ub = target.get("upper_z-slice_limit")
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

            if particle_id not in results:
                results[particle_id] = {
                    "Recalls": [float(recall)],
                    "Average recall": 0.0,
                    "Standard deviation on recall": 0.0,
                }
            else:
                results[particle_id]["Recalls"].append(recall)

    for particle_id in results:
        results[particle_id]["Average recall"] = float(
            np.mean(np.array(results[particle_id]["Recalls"]))
        )
        results[particle_id]["Standard deviation on recall"] = float(
            np.std(np.array(results[particle_id]["Recalls"]))
        )

    with open(
        os.path.join(
            f"/home/shreyas/Dropbox/miningTomograms/particlewise_recall/{dataset_name}/raw/",
            f"particlewise_recall_{feature_extraction_method}_{clustering_method}_{particle_extraction_method}.yaml",
        ),
        "w",
    ) as outf:
        yaml.dump(results, outf)


if __name__ == "__main__":
    main()
