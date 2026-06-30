import os
import yaml
import starfile
import ndjson
import numpy as np

from scipy.spatial.distance import cdist
from scipy import stats

# ==================================================
# READ DATASET TYPE
# ==================================================

def get_threshold_angstrom(dataset_type):

    if dataset_type in ["tutorial", "real"]:

        return 125

    elif dataset_type == ("simulated"):

        return 100

    else:

        raise ValueError(f"Unknown dataset type: {dataset_type}")

# ==================================================
# READ GROUND TRUTH NDJSON
# ==================================================

def read_ndjson_coords(fname):

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


# ==================================================
# READ PREDICTION YAML
# ==================================================

def read_prediction_star(fname, tomogram_shape, voxel_size):

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


# ==================================================
# ANGSTROM -> VOXEL THRESHOLD
# ==================================================

def get_voxel_threshold(angstrom_threshold, voxel_size):

    return int(round(angstrom_threshold / voxel_size, 0))


# ==================================================
# PICKET-STYLE METRICS
# ==================================================

def compute_metrics(distances, threshold, num_predictions, num_ground_truth):

    masked = np.where(distances <= threshold, 1, 0)

    positive_prediction_idxs = np.any(masked, axis=1)

    captured_ground_truth_idxs = np.any(masked, axis=0)

    precision = 0.0
    recall = 0.0

    if num_predictions > 0:

        precision = (np.count_nonzero(positive_prediction_idxs) / num_predictions)

    if num_ground_truth > 0:

        recall = (np.count_nonzero(captured_ground_truth_idxs) / num_ground_truth)

    if precision == 0 or recall == 0:

        f1 = 0.0

    else:

        f1 = float(stats.hmean([precision, recall]))

    return (float(precision), float(recall), float(f1))

# ==================================================
# MAIN EVALUATION
# ==================================================

def run_evaluation(prediction_star,
    gt_ndjson,
    threshold_angstrom,
    output_yaml,
    tomogram_shape,
    voxel_size
):
    
    if not os.path.exists(prediction_star):
        print(f"Missing prediction file: {prediction_star}")
        return

    print("\nRunning Evaluation...\n")

    # ----------------------------------------------
    # Read predictions
    # ----------------------------------------------

    pred_coords = read_prediction_star(prediction_star, tomogram_shape, voxel_size)

    if len(pred_coords) == 0:
        print("No predicted particles.")
        results = {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "num_predictions": 0.0
        }

        with open(output_yaml, "w") as f:
            yaml.dump(results, f, sort_keys=False)

        return

    # ----------------------------------------------
    # Read GT
    # ----------------------------------------------

    gt_coords = read_ndjson_coords(gt_ndjson)

    voxel_threshold = get_voxel_threshold(threshold_angstrom, voxel_size)

    # ----------------------------------------------
    # Distance matrix
    # ----------------------------------------------

    distances = cdist(pred_coords, gt_coords, metric="euclidean")

    precision, recall, f1 = compute_metrics(distances, voxel_threshold, len(pred_coords), len(gt_coords))

    results = {

        "precision": precision,

        "recall": recall,

        "f1_score": f1,

        "num_predictions": int(len(pred_coords)),

        "num_ground_truth": int(len(gt_coords)),

        "threshold_angstrom": threshold_angstrom,

        "voxel_threshold": voxel_threshold

    }

    with open(output_yaml, "w") as f:

        yaml.dump(results, f, sort_keys=False)

    print("\nEvaluation Results:\n")

    print(yaml.dump(results,sort_keys=False))

    print(f"Saved: {output_yaml}")
