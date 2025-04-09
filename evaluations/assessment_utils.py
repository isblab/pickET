import mrcfile
import numpy as np


def get_tomo_shape(fname: str) -> tuple[int, int, int]:
    with mrcfile.open(fname, mode="r", permissive=True) as mrcf:
        tomo = np.array(mrcf.data)
    return tuple(tomo.shape)


def get_voxel_threshold(fname: str, angs_threshold: float) -> int:
    with mrcfile.open(fname, mode="r", permissive=True) as mrcf:
        voxel_sizes = np.array(
            [mrcf.voxel_size.z, mrcf.voxel_size.y, mrcf.voxel_size.x], dtype=np.float32
        )

    voxel_size = np.max(voxel_sizes)
    threshold = int(round(angs_threshold / voxel_size, 0))
    return threshold


def zslice_filter_ground_truth_centroids(
    ground_truth_centroids: np.ndarray, zslice_lb: int, zslice_ub: int
) -> np.ndarray:
    c1 = zslice_lb is None or ground_truth_centroids[:, 0] >= zslice_lb
    c2 = zslice_ub is None or ground_truth_centroids[:, 0] <= zslice_ub
    out_centroids = ground_truth_centroids[c1 & c2].squeeze()
    return out_centroids


def get_random_centroids(
    tomo_shape: tuple[int, int, int], num_centroids: int
) -> np.ndarray:
    random_centroids = []
    while len(random_centroids) < num_centroids:
        rc = (
            np.random.randint(0, tomo_shape[0]),
            np.random.randint(0, tomo_shape[1]),
            np.random.randint(0, tomo_shape[2]),
        )
        if rc not in random_centroids:
            random_centroids.append(rc)

    return np.array(random_centroids)


def compute_precision_recall_f1score(
    distances: np.ndarray, threshold: float, num_pred: int, num_true: int
) -> tuple[float, float, float]:
    precision, recall, f1_score = 0.0, 0.0, 0.0

    masked = np.where(distances <= threshold, 1, 0)
    positive_pred_idxs = np.any(masked, axis=1)
    captured_particle_idxs = np.any(masked, axis=0)

    if num_pred > 0:
        precision = np.count_nonzero(positive_pred_idxs) / num_pred
    if num_true > 0:
        recall = np.count_nonzero(captured_particle_idxs) / num_true
    if precision + recall != 0:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return float(precision), float(recall), float(f1_score)


def compute_global_metrics(
    results: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    precisions, recalls, f1scores = [], [], []
    random_precisions, random_recalls, random_f1scores = [], [], []
    for tomo in results:
        if not tomo.startswith("Tomogram_"):
            continue
        precisions.append(results[tomo]["Precision"])
        recalls.append(results[tomo]["Recall"])
        f1scores.append(results[tomo]["F1-score"])
        random_precisions.append(results[tomo]["Random Precision"])
        random_recalls.append(results[tomo]["Random Recall"])
        random_f1scores.append(results[tomo]["Random F1-score"])

    glob_prec = float(np.mean(np.array(precisions)))
    glob_recall = float(np.mean(np.array(recalls)))
    glob_f1score = float(np.mean(np.array(f1scores)))
    glob_random_prec = float(np.mean(np.array(random_precisions)))
    glob_random_recall = float(np.mean(np.array(random_recalls)))
    glob_random_f1score = float(np.mean(np.array(random_f1scores)))
    results["Global"] = {
        "Precision": glob_prec,
        "Recall": glob_recall,
        "F1-score": glob_f1score,
        "Random Precision": glob_random_prec,
        "Random Recall": glob_random_recall,
        "Random F1-score": glob_random_f1score,
    }

    return results
