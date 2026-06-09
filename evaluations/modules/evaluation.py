import yaml
import ndjson
import numpy as np

from scipy.spatial.distance import cdist
from scipy import stats

# ==================================================
# READ DATASET TYPE
# ==================================================

def get_threshold_angstrom(
    dataset_type
):

    if dataset_type in [
        "tutorial",
        "real"
    ]:

        return 125

    elif dataset_type == (
        "simulated"
    ):

        return 100

    else:

        raise ValueError(
            f"Unknown dataset type: "
            f"{dataset_type}"
        )

# ==================================================
# READ GROUND TRUTH NDJSON
# ==================================================

def read_ndjson_coords(
    fname
):

    with open(
        fname,
        "r"
    ) as f:

        annotations = ndjson.load(f)

    coords = np.zeros(

        (len(annotations), 3),

        dtype=np.int32

    )

    for idx, ann in enumerate(
        annotations
    ):

        coords[idx] = [

            ann["location"]["z"],

            ann["location"]["y"],

            ann["location"]["x"]

        ]

    return coords


# ==================================================
# READ PREDICTION YAML
# ==================================================

def read_prediction_yaml(
    fname
):

    with open(
        fname,
        "r"
    ) as f:

        data = yaml.safe_load(f)

    coords = []

    for particle in data[
        "Predicted_Particle_Centroid_Coordinates"
    ]:

        coords.append([

            particle["z"],

            particle["y"],

            particle["x"]

        ])

    return np.array(
        coords
    )


# ==================================================
# ANGSTROM -> VOXEL THRESHOLD
# ==================================================

def get_voxel_threshold(

    angstrom_threshold,

    voxel_size

):

    return int(

        round(

            angstrom_threshold
            /
            voxel_size,

            0

        )

    )


# ==================================================
# PICKET-STYLE METRICS
# ==================================================

def compute_metrics(

    distances,

    threshold,

    num_predictions,

    num_ground_truth

):

    masked = np.where(

        distances <= threshold,

        1,

        0

    )

    positive_prediction_idxs = np.any(

        masked,

        axis=1

    )

    captured_ground_truth_idxs = np.any(

        masked,

        axis=0

    )

    precision = 0.0
    recall = 0.0

    if num_predictions > 0:

        precision = (

            np.count_nonzero(

                positive_prediction_idxs

            )
            /
            num_predictions

        )

    if num_ground_truth > 0:

        recall = (

            np.count_nonzero(

                captured_ground_truth_idxs

            )
            /
            num_ground_truth

        )

    if precision == 0 or recall == 0:

        f1 = 0.0

    else:

        f1 = float(

            stats.hmean(

                [precision, recall]

            )

        )

    return (

        float(precision),

        float(recall),

        float(f1)

    )


# ==================================================
# MAIN EVALUATION
# ==================================================

def run_evaluation(

    prediction_yaml,

    gt_ndjson,

    threshold_angstrom,

    output_yaml

):

    print(
        "\nRunning Evaluation...\n"
    )

    # ----------------------------------------------
    # Read predictions
    # ----------------------------------------------

    pred_coords = read_prediction_yaml(

        prediction_yaml

    )

    # ----------------------------------------------
    # Read GT
    # ----------------------------------------------

    gt_coords = read_ndjson_coords(

        gt_ndjson

    )

    # ----------------------------------------------
    # Read metadata
    # ----------------------------------------------

    with open(
        prediction_yaml,
        "r"
    ) as f:

        pred_yaml = yaml.safe_load(f)

    voxel_size = float(

        pred_yaml["metadata"][
            "voxel_size"
        ]

    )

    voxel_threshold = get_voxel_threshold(

        threshold_angstrom,

        voxel_size

    )

    # ----------------------------------------------
    # Distance matrix
    # ----------------------------------------------

    distances = cdist(

        pred_coords,

        gt_coords,

        metric="euclidean"

    )

    precision, recall, f1 = compute_metrics(

        distances,

        voxel_threshold,

        len(pred_coords),

        len(gt_coords)

    )

    results = {

        "precision": precision,

        "recall": recall,

        "f1_score": f1,

        "num_predictions": int(
            len(pred_coords)
        ),

        "num_ground_truth": int(
            len(gt_coords)
        ),

        "threshold_angstrom":
            threshold_angstrom,

        "voxel_threshold":
            voxel_threshold

    }

    with open(
        output_yaml,
        "w"
    ) as f:

        yaml.dump(

            results,

            f,

            sort_keys=False

        )

    print(
        "\nEvaluation Results:\n"
    )

    print(

        yaml.dump(

            results,

            sort_keys=False

        )

    )

    print(
        f"Saved: {output_yaml}"
    )
