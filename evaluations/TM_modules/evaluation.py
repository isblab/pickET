import os
import yaml
import starfile
import ndjson
import numpy as np

from scipy.spatial.distance import cdist
from scipy import stats


def get_threshold_angstrom(dataset_type: str) -> float:
    """ Get a threshold to consider predicted particle
    as a match to GT

    Args:
        dataset_type (str): Type of dataset
            "simulated" or "experimental" or "tutorial".

    Returns:
        float: Threshold in angstrom
    """

    if dataset_type in ["tutorial", "real"]:

        return 125

    elif dataset_type == ("simulated"):

        return 100

    else:

        raise ValueError(f"Unknown dataset type: {dataset_type}")


def read_ndjson_coords(fname: str) -> np.ndarray:
    """ Read ground truth ndjson file.

    Args:
        fname (str): Path to ground truth ndjson file

    Returns:
        np.ndarray: Z, Y, X coordinates of the ground truth particles
    """

    with open(fname, "r") as f:

        annotations = ndjson.load(f)

    coords = np.zeros((len(annotations), 3), dtype=np.int32)

    for idx, ann in enumerate(annotations):

        coords[idx] = [
            ann["location"]["z"],
            ann["location"]["y"],
            ann["location"]["x"]
        ]

    return coords


def read_prediction_star(
    fname: str,
    tomogram_shape: tuple,
    voxel_size: float,
) -> np.ndarray:
    """ Read <>_prediction.star file

    Args:
        fname (str): Path of <>_prediction.star
        tomogram_shape (tuple): Shape of the tomogram (X, Y, Z)
        voxel_size (float): Voxel size in angstrom

    Returns:
        np.ndarray: Z, Y, X coordinates of predicted particles
    """

    df = starfile.read(fname)

    coords = []

    for _, row in df.iterrows():

        if "rlnCoordinateX" in row.index:

            x = int(round(float(row["rlnCoordinateX"])))
            y = int(round(float(row["rlnCoordinateY"])))
            z = int(round(float(row["rlnCoordinateZ"])))

        elif "rlnCenteredCoordinateXAngst" in row.index:

            x = int(round(float(row["rlnCenteredCoordinateXAngst"]) / voxel_size))
            y = int(round(float(row["rlnCenteredCoordinateYAngst"]) / voxel_size))
            z = int(round(float(row["rlnCenteredCoordinateZAngst"]) / voxel_size))

            x += tomogram_shape[2] / 2
            y += tomogram_shape[1] / 2
            z += tomogram_shape[0] / 2

            x = int(round(x))
            y = int(round(y))
            z = int(round(z))

        else:

            raise ValueError("No coordinate columns found.")

        coords.append([z, y, x])

    return np.array(coords)


def get_voxel_threshold(
    threshold_angstrom: float,
    voxel_size: float
):
    """ Get a voxel threshold to consider predicted particle
    as a match to GT

    Args:
        dataset_type (str): Type of dataset
            "simulated" or "experimental" or "tutorial".

    Args:
        threshold_angstrom (float): Threshold in angstroms
        voxel_size (float): Voxel size in angstroms

    Returns:
        int: Voxel threshold
    """

    return int(round(threshold_angstrom / voxel_size, 0))


def compute_metrics(
    distances: np.ndarray,
    voxel_threshold: float,
    num_predictions: int,
    num_ground_truth: int,
):
    """ Compute PickET-style metrics

    Args:
        distances (np.ndarray): Distances between GT and predicted particles
        voxel_threshold (float): Distance threshold between GT and predicted particle (in voxels)
        num_predictions (int): Total predicted particles
        num_ground_truth (int): Total GT particles

    Returns:
        tuple: A tuple of precision, recall, f1-score
    """

    masked = np.where(distances <= voxel_threshold, 1, 0)

    positive_prediction_idxs = np.any(masked, axis=1)

    captured_ground_truth_idxs = np.any(masked, axis=0)

    precision = 0.0
    recall = 0.0

    if num_predictions > 0:

        precision = np.count_nonzero(positive_prediction_idxs) / num_predictions

    if num_ground_truth > 0:

        recall = np.count_nonzero(captured_ground_truth_idxs) / num_ground_truth

    if precision == 0 or recall == 0:

        f1 = 0.0

    else:

        f1 = float(stats.hmean([precision, recall]))

    return (float(precision), float(recall), float(f1))


def run_evaluation(
    prediction_star: str,
    gt_ndjson: str,
    threshold_angstrom: float,
    output_yaml: str,
    tomogram_shape: tuple,
    voxel_size: float,
):
    """ Main evaluation

    Args:
        prediction_star (str): Path of <>_prediction.star
        gt_ndjson (str): Path to ground truth ndjson file
        threshold_angstrom (float): Threshold in angstroms
        output_yaml (str): Path to yaml file to save evalutation
        tomogram_shape (tuple): Shape of the tomogram (X, Y, Z)
        voxel_size (float): Voxel size in angstrom
    """

    if not os.path.exists(prediction_star):
        print(f"Missing prediction file: {prediction_star}")
        return

    print("\nRunning Evaluation...\n")

    pred_coords = read_prediction_star(
        fname=prediction_star,
        tomogram_shape=tomogram_shape,
        voxel_size=voxel_size,
    )

    gt_coords = read_ndjson_coords(fname=gt_ndjson)

    if len(pred_coords) == 0:
        print("No predicted particles.")
        results = {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "num_predictions": 0.0,
            "num_ground_truth": int(len(gt_coords)),
        }

        with open(output_yaml, "w") as f:
            yaml.dump(results, f, sort_keys=False)

        return

    voxel_threshold = get_voxel_threshold(
        threshold_angstrom=threshold_angstrom,
        voxel_size=voxel_size,
    )

    distances = cdist(pred_coords, gt_coords, metric="euclidean")

    precision, recall, f1 = compute_metrics(
        distances=distances,
        voxel_threshold=voxel_threshold,
        num_predictions=len(pred_coords),
        num_ground_truth=len(gt_coords),
    )

    results = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "num_predictions": int(len(pred_coords)),
        "num_ground_truth": int(len(gt_coords)),
        "threshold_angstrom": threshold_angstrom,
        "voxel_threshold": voxel_threshold,
    }

    with open(output_yaml, "w") as f:

        yaml.dump(results, f, sort_keys=False)

    print("\nEvaluation Results:\n")

    print(yaml.dump(results, sort_keys=False))

    print(f"Saved: {output_yaml}")
