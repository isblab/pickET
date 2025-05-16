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
    parent_path = params["parent_path"]
    output_dir = os.path.join(params["output_dir"], dataset_name, "evaluations")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    groups = assessment_utils.separate_files_into_groups(parent_path)
    for group_name, pc_fname in groups.items():
        results = {}
        ### Processing block
        for idx, target in enumerate(pc_fname):
            pred_centroids, pred_metadata = assessment_utils.load_predictions(target)
            zslice_lb = pred_metadata["z_lb_for_particle_extraction"]
            zslice_ub = pred_metadata["z_ub_for_particle_extraction"]
            tomo_shape = tuple(np.array(pred_metadata["tomogram_shape"]).tolist())
            gt_fpath = assessment_utils.get_ground_truth_fpath(
                str(pred_metadata["tomogram_path"])
            )

            threshold = assessment_utils.get_voxel_threshold(
                angstrom_threshold, pred_metadata["voxel_size"]
            )
            pred_centroids = utils.read_yaml_coords(target)
            print(target, len(pred_centroids), sep=":\t")

            if len(pred_centroids) <= 1_000_000:
                random_centroids = assessment_utils.get_random_centroids(
                    tomo_shape, num_centroids=len(pred_centroids)
                )
                true_centroids = assessment_utils.read_ndjson_coords(gt_fpath)
                true_centroids = assessment_utils.zslice_filter_ground_truth_centroids(
                    true_centroids, zslice_lb, zslice_ub
                )

                n_pred = len(pred_centroids)
                n_true = len(true_centroids)
                n_rand = len(random_centroids)

                distances = cdist(pred_centroids, true_centroids, metric="euclidean")
                distances_random = cdist(
                    random_centroids, true_centroids, metric="euclidean"
                )

                precision, recall, f1score = (
                    assessment_utils.compute_precision_recall_f1score(
                        distances, threshold, num_pred=n_pred, num_true=n_true
                    )
                )
                random_precision, random_recall, random_f1score = (
                    assessment_utils.compute_precision_recall_f1score(
                        distances_random, threshold, num_pred=n_rand, num_true=n_true
                    )
                )

                results[f"Tomo_ID - {idx}"] = {
                    "Precision": precision,
                    "Recall": recall,
                    "F1-score": f1score,
                    "Random Precision": random_precision,
                    "Random Recall": random_recall,
                    "Random F1-score": random_f1score,
                }

            else:  # Ignore the tomogram where the number of predicted particles is over 1M
                results[f"Tomo_ID - {idx}"] = {
                    "Precision": np.nan,
                    "Recall": np.nan,
                    "F1-score": np.nan,
                    "Random Precision": np.nan,
                    "Random Recall": np.nan,
                    "Random F1-score": np.nan,
                }

        results = assessment_utils.compute_global_metrics(results)

        with open(
            os.path.join(
                output_dir,
                f"overall_results_{group_name}.yaml",  # type:ignore
            ),
            "w",
        ) as outf:
            yaml.dump(results, outf)
        print()


if __name__ == "__main__":
    main()
