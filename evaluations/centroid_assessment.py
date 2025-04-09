import os
import sys
import yaml
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist

from assets import utils
import assessment_utils


def main():
    ### Input block
    params_fname = sys.argv[1]

    params = utils.load_params_from_yaml(params_fname)

    experiment_name = params["experiment_name"]
    run_name = params["run_name"]
    angstrom_threshold = params["threshold_in_angstrom"]
    inputs = params["inputs"]
    clustering_method = params["clustering_method"]
    particle_extraction_method = params["particle_extraction_method"]
    output_dir = os.path.join(params["output_dir"], experiment_name, run_name)

    results = {}
    ### Processing block
    for idx, target in enumerate(tqdm(inputs)):
        threshold = assessment_utils.get_voxel_threshold(
            target["tomogram"], angs_threshold=angstrom_threshold
        )
        zslice_lb = target.get("lower_z-slice_limit")
        zslice_ub = target.get("upper_z-slice_limit")

        tomo_shape = assessment_utils.get_tomo_shape(target["tomogram"])

        pred_centroids = utils.read_ndjson_coords(target["predicted_centroids"])
        random_centroids = assessment_utils.get_random_centroids(
            tomo_shape, num_centroids=len(pred_centroids)
        )
        true_centroids = utils.read_ndjson_coords(target["true_centroids"])
        true_centroids = assessment_utils.zslice_filter_ground_truth_centroids(
            true_centroids, zslice_lb, zslice_ub
        )

        n_pred = len(pred_centroids)
        n_true = len(true_centroids)
        n_rand = len(random_centroids)

        distances = cdist(pred_centroids, true_centroids, metric="euclidean")
        distances_random = cdist(random_centroids, true_centroids, metric="euclidean")

        precision, recall, f1score = assessment_utils.compute_precision_recall_f1score(
            distances, threshold, num_pred=n_pred, num_true=n_true
        )
        random_precision, random_recall, random_f1score = (
            assessment_utils.compute_precision_recall_f1score(
                distances_random, threshold, num_pred=n_rand, num_true=n_true
            )
        )

        results[f"Tomogram_{idx}"] = {
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1score,
            "Random Precision": random_precision,
            "Random Recall": random_recall,
            "Random F1-score": random_f1score,
        }

    results = assessment_utils.compute_global_metrics(results)

    with open(
        os.path.join(
            output_dir,
            f"overall_results_{clustering_method}_{particle_extraction_method}.yaml",
        ),
        "w",
    ) as outf:
        yaml.dump(results, outf)


if __name__ == "__main__":
    main()
