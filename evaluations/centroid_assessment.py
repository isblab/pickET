import os
import sys
import yaml
import numpy as np
from rich.progress import track
from scipy.spatial.distance import cdist

from picket.core import utils
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
        for idx, target in enumerate(
            track(pc_fname, description=f"Processing {group_name}")
        ):
            pred_centroids, pred_metadata = utils.load_predictions(target)
            if len(pred_centroids) < 1_000_000:
                zslice_lb = pred_metadata.get("z_lb_for_particle_extraction", "None")
                zslice_ub = pred_metadata.get("z_ub_for_particle_extraction", "None")

                tomo_shape = tuple(np.array(pred_metadata["tomogram_shape"]).tolist())
                gt_fpath = assessment_utils.get_ground_truth_fpath(
                    pred_metadata["tomogram_path"],
                    annot_dir_head=params["annot_dir_head"],
                )

                threshold = assessment_utils.get_voxel_threshold(
                    angstrom_threshold, pred_metadata["voxel_size"]
                )

                true_centroids = assessment_utils.read_ndjson_coords(gt_fpath)
                true_centroids = assessment_utils.zslice_filter_ground_truth_centroids(
                    true_centroids, zslice_lb, zslice_ub
                )
                gtr_centroids = assessment_utils.get_random_centroids(
                    tomo_shape, num_centroids=len(true_centroids)
                )
                mdr_centroids = assessment_utils.get_random_centroids(
                    tomo_shape, num_centroids=len(pred_centroids)
                )

                n_pred = len(pred_centroids)
                n_true = len(true_centroids)
                n_gtr = len(gtr_centroids)
                n_mdr = len(mdr_centroids)

                if n_true != 0 and n_pred != 0:
                    distances = cdist(
                        pred_centroids, true_centroids, metric="euclidean"
                    )
                    distances_gtr = cdist(
                        gtr_centroids, true_centroids, metric="euclidean"
                    )
                    distances_mdr = cdist(
                        mdr_centroids, true_centroids, metric="euclidean"
                    )

                    precision, recall, f1score = (
                        assessment_utils.compute_precision_recall_f1score(
                            distances, threshold, num_pred=n_pred, num_true=n_true
                        )
                    )
                    gtr_precision, gtr_recall, gtr_f1score = (
                        assessment_utils.compute_precision_recall_f1score(
                            distances_gtr, threshold, num_pred=n_gtr, num_true=n_true
                        )
                    )
                    mdr_precision, mdr_recall, mdr_f1score = (
                        assessment_utils.compute_precision_recall_f1score(
                            distances_mdr, threshold, num_pred=n_mdr, num_true=n_true
                        )
                    )
                    relative_recall = assessment_utils.compute_relative_recall(
                        recall, mdr_random_recall=mdr_recall
                    )
                else:
                    precision, recall, f1score = np.nan, np.nan, np.nan
                    gtr_precision, gtr_recall, gtr_f1score = np.nan, np.nan, np.nan
                    mdr_precision, mdr_recall, mdr_f1score = np.nan, np.nan, np.nan
                    relative_recall = np.nan

            else:
                precision, recall, f1score = np.nan, np.nan, np.nan
                gtr_precision, gtr_recall, gtr_f1score = np.nan, np.nan, np.nan
                mdr_precision, mdr_recall, mdr_f1score = np.nan, np.nan, np.nan
                relative_recall = np.nan
                n_pred, n_true = np.nan, np.nan

            results[f"Tomo_ID - {idx}"] = {
                "Precision": precision,
                "Recall": recall,
                "F1-score": f1score,
                "GTR Precision": gtr_precision,
                "GTR Recall": gtr_recall,
                "GTR F1-score": gtr_f1score,
                "MDR Precision": mdr_precision,
                "MDR Recall": mdr_recall,
                "MDR F1-score": mdr_f1score,
                "Relative recall": relative_recall,
                "Number of predicted particles": n_pred,
                "Number of ground truth annotations": n_true,
                "Time taken for S1": pred_metadata.get("time_taken_for_s1", np.nan),
                "Time taken for S2": pred_metadata.get("time_taken_for_s2", np.nan),
                "Total time taken": pred_metadata.get("time_taken_for_s1", np.nan)
                + pred_metadata.get("time_taken_for_s2", np.nan),
            }

        # results = assessment_utils.compute_global_metrics(results)

        with open(
            os.path.join(
                output_dir,
                f"overall_results_{group_name}.yaml",  # type:ignore
            ),
            "w",
        ) as outf:
            yaml.dump(results, outf)


if __name__ == "__main__":
    main()
