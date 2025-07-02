import os
import glob
import yaml
import ndjson
import numpy as np


def separate_files_into_groups(parent_fpath: str) -> dict:
    all_fpath = sorted(glob.glob(os.path.join(parent_fpath, "*.yaml")))
    out_groups = {}

    for fpath in all_fpath:
        fname = fpath.split("/")[-1]
        fex_mode = fname.split("_")[3]
        cl_mode = fname.split("_")[4]
        f2 = fname[:-5]
        pex_mode = "_".join(f2.split("_")[5:])

        dictkey = f"{fex_mode}_{cl_mode}_{pex_mode}"
        if dictkey not in out_groups:
            out_groups[dictkey] = [fpath]
        else:
            out_groups[dictkey].append(fpath)

    return out_groups


def get_ground_truth_fpath(tomogram_path: str, annot_dir_head: str) -> str:
    tomogram_dir = "/".join(tomogram_path.split("/")[:-1])
    annot_path = os.path.join(tomogram_dir, annot_dir_head, "all_annotations.ndjson")
    return annot_path


def read_ndjson_coords(fname: str) -> np.ndarray:
    with open(fname, "r") as in_annot_f:
        annotations = ndjson.load(in_annot_f)

    coords = np.nan * np.ones((len(annotations), 3), dtype=np.int32)
    for idx, ln in enumerate(annotations):
        coords[idx] = np.array(
            [ln["location"]["z"], ln["location"]["y"], ln["location"]["x"]]
        )

    if np.any(np.isnan(coords)):
        raise ValueError("Something went wrong when reading coords")

    return coords


def load_predictions(pred_fname: str) -> tuple[list[dict], dict]:
    with open(pred_fname, "r") as predf:
        contents = yaml.safe_load(predf)
        pred_coords = contents["Predicted_Particle_Centroid_Coordinates"]
        metadata = contents["metadata"]
    return pred_coords, metadata


def get_voxel_threshold(angs_threshold: float, voxel_sizes: list) -> int:
    voxel_size = np.max(np.array(voxel_sizes))
    threshold = int(round(angs_threshold / voxel_size, 0))
    return threshold


def zslice_filter_ground_truth_centroids(
    ground_truth_centroids: np.ndarray, zslice_lb, zslice_ub
) -> np.ndarray:
    if zslice_lb != "None":
        ground_truth_centroids = ground_truth_centroids[
            np.where(ground_truth_centroids[:, 0] >= int(zslice_lb))
        ]
    if zslice_ub != "None":
        ground_truth_centroids = ground_truth_centroids[
            np.where(ground_truth_centroids[:, 0] <= int(zslice_ub))
        ]
    return ground_truth_centroids


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
    gtr_precisions, gtr_recalls, gtr_f1scores = [], [], []
    mdr_precisions, mdr_recalls, mdr_f1scores = [], [], []
    for tomo in results:
        if not tomo.startswith("Tomo_"):
            continue
        if results[tomo]["Precision"] == np.nan:
            continue
        precisions.append(results[tomo]["Precision"])
        recalls.append(results[tomo]["Recall"])
        f1scores.append(results[tomo]["F1-score"])
        gtr_precisions.append(results[tomo]["GTR Precision"])
        gtr_recalls.append(results[tomo]["GTR Recall"])
        gtr_f1scores.append(results[tomo]["GTR F1-score"])
        mdr_precisions.append(results[tomo]["MDR Precision"])
        mdr_recalls.append(results[tomo]["MDR Recall"])
        mdr_f1scores.append(results[tomo]["MDR F1-score"])

    glob_prec = float(np.mean(np.array(precisions)))
    glob_recall = float(np.mean(np.array(recalls)))
    glob_f1score = float(np.mean(np.array(f1scores)))
    glob_gtr_prec = float(np.mean(np.array(gtr_precisions)))
    glob_gtr_recall = float(np.mean(np.array(gtr_recalls)))
    glob_gtr_f1score = float(np.mean(np.array(gtr_f1scores)))
    glob_mdr_prec = float(np.mean(np.array(mdr_precisions)))
    glob_mdr_recall = float(np.mean(np.array(mdr_recalls)))
    glob_mdr_f1score = float(np.mean(np.array(mdr_f1scores)))

    results["Global"] = {
        "Precision": glob_prec,
        "Recall": glob_recall,
        "F1-score": glob_f1score,
        "GTR Precision": glob_gtr_prec,
        "GTR Recall": glob_gtr_recall,
        "GTR F1-score": glob_gtr_f1score,
        "MDR Precision": glob_mdr_prec,
        "MDR Recall": glob_mdr_recall,
        "MDR F1-score": glob_mdr_f1score,
    }

    return results
