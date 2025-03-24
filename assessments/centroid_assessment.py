import os
import sys
import yaml
import mrcfile
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist

from assets import utils


def get_voxel_threshold(fname: str, angs_threshold: float) -> int:
    with mrcfile.open(fname, mode="r", permissive=True) as mrcf:
        voxel_sizes = np.array(
            [mrcf.voxel_size.z, mrcf.voxel_size.y, mrcf.voxel_size.x], dtype=np.float32
        )

    voxel_size = np.max(voxel_sizes)
    threshold = int(round(angs_threshold / voxel_size, 0))
    return threshold


def compute_precision_recall_f1score(
    distances: np.ndarray, threshold: float, num_pred: int, num_true: int
) -> tuple[float, float, float]:
    precision, recall, f1_score = 0.0, 0.0, 0.0

    masked = np.where(distances <= threshold, 1, 0)
    tp_count = np.sum(np.any(masked == 1, axis=1))

    if num_pred > 0:
        precision = tp_count / num_pred
    if num_true > 0:
        recall = tp_count / num_true
    if num_true + num_pred > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return float(precision), float(recall), float(f1_score)


def compute_global_metrics(
    results: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    precisions, recalls, f1scores = [], [], []
    for tomo in results:
        if not tomo.startswith("Tomogram_"):
            continue
        precisions.append(results[tomo]["Precision"])
        recalls.append(results[tomo]["Recall"])
        f1scores.append(results[tomo]["F1-score"])

    glob_prec = float(np.mean(np.array(precisions)))
    glob_recall = float(np.mean(np.array(recalls)))
    glob_f1score = float(np.mean(np.array(f1scores)))
    results["Global"] = {
        "Precision": glob_prec,
        "Recall": glob_recall,
        "F1-score": glob_f1score,
    }

    return results


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
        # print(f"Processing prediction {idx+1}/{len(inputs)}")
        threshold = get_voxel_threshold(
            target["tomogram"], angs_threshold=angstrom_threshold
        )
        pred_centroids = utils.read_ndjson_coords(target["predicted_centroids"])
        true_centroids = utils.read_ndjson_coords(target["true_centroids"])
        n_pred = len(pred_centroids)
        n_true = len(true_centroids)

        distances = cdist(pred_centroids, true_centroids, metric="euclidean")
        precision, recall, f1score = compute_precision_recall_f1score(
            distances, threshold, num_pred=n_pred, num_true=n_true
        )

        results[f"Tomogram_{idx}"] = {
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1score,
        }

    results = compute_global_metrics(results)

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
