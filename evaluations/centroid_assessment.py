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

    dataset_name = params["dataset_name"]
    angstrom_threshold = params["threshold_in_angstrom"]
    inputs = params["inputs"]
    output_dir = os.path.join(params["output_dir"], dataset_name, "evaluations")

    results = {}
    ### Processing block
    for idx, target in enumerate(tqdm(inputs)):
        tomo_shape = assessment_utils.get_tomo_shape(
            assessment_utils.get_metadata_entry(
                target["predicted_centroids"], "tomogram_path"
            )
        )
        threshold = assessment_utils.get_voxel_threshold(
            angstrom_threshold,
            assessment_utils.get_metadata_entry(
                target["predicted_centroids"], "voxel_size"
            ),
        )

        zslice_lb = assessment_utils.get_metadata_entry(
            target["predicted_centroids"], "z_lb_for_particle_extraction"
        )
        zslice_ub = assessment_utils.get_metadata_entry(
            target["predicted_centroids"], "z_ub_for_particle_extraction"
        )

        clustering_method = str(
            assessment_utils.get_metadata_entry(
                target["predicted_centroids"], "clustering_method"
            )
        )
        particle_extraction_method = str(
            assessment_utils.get_metadata_entry(
                target["predicted_centroids"], "pex_mode"
            )
        )

        pred_centroids = utils.read_yaml_coords(target["predicted_centroids"])
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
            f"overall_results_{clustering_method}_{particle_extraction_method}.yaml",  # type:ignore
        ),
        "w",
    ) as outf:
        yaml.dump(results, outf)


if __name__ == "__main__":
    main()
