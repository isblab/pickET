import mrcfile
import numpy as np


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
    positive_pred_idxs = np.any(masked, axis=1)
    captured_particle_idxs = np.any(masked, axis=0)

    if num_pred > 0:
        precision = np.count_nonzero(positive_pred_idxs) / num_pred
    if num_true > 0:
        recall = np.count_nonzero(captured_particle_idxs) / num_true
    if precision + recall != 0:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return float(precision), float(recall), float(f1_score)
