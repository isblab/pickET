import yaml
import numpy as np
from typing import Optional
from rich.progress import track
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import assessment_utils


def get_euclidean_distance(p1, p2):
    d2 = ((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2)
    return float(np.sqrt(d2))


def get_random_coords(
    ncoords: int,
    shape: tuple[int, int],
    enfore_fp: bool = False,
    gt_coords: Optional[list[list[int]]] = None,
    threshold: Optional[float] = None,
):
    coords = []
    while len(coords) < ncoords:
        y = np.random.uniform(0, shape[0], ncoords).astype(np.int32)
        x = np.random.uniform(0, shape[1], ncoords).astype(np.int32)
        for yi, xi in zip(y, x):
            if enfore_fp:
                if gt_coords is None or threshold is None:
                    raise TypeError()

                gt_coords_arr = np.array(gt_coords)
                proposal_coord = np.array([yi, xi])
                distances = cdist(np.expand_dims(proposal_coord, axis=0), gt_coords_arr)
                masked_distances = np.where(distances <= float(threshold), 1, 0)
                if np.any(masked_distances, axis=1)[0]:
                    continue

            coords.append([yi, xi])
    return coords


def simulate_a_model(
    ncoords: int,
    prop_tp: float,
    shape: tuple[int, int],
    gt_coords: list[list[int]],
    threshold: int,
):
    coords = []
    tp_count = int(round(ncoords * prop_tp, 0))
    while len(coords) < tp_count:
        i = np.random.randint(0, len(gt_coords))
        base_coord = gt_coords[i]
        c = [
            np.random.randint(
                base_coord[0] - threshold // 2, base_coord[0] + (threshold // 2) + 1
            ),
            np.random.randint(
                base_coord[1] - threshold // 2, base_coord[1] + (threshold // 2) + 1
            ),
        ]
        if c[0] >= 0 and c[0] <= shape[0] and c[1] >= 0 and c[1] <= shape[1]:
            if get_euclidean_distance(base_coord, c) <= threshold:
                coords.append(c)

    fp_coords = get_random_coords(
        ncoords - tp_count,
        shape,
        enfore_fp=True,
        gt_coords=gt_coords,
        threshold=threshold,
    )
    coords.extend(fp_coords)
    return coords


def get_color_and_marker_mappings(results, colors, markers):
    # num_preds is represented by colors and prop_tp is represented by markers
    unique_num_preds = np.unique(results[:, 0])
    unique_prop_tps = np.unique(results[:, 1])

    color_mapping, marker_mapping = {}, {}
    for i, j in zip(unique_num_preds, colors):
        color_mapping[np.int32(i)] = j
    for i, j in zip(unique_prop_tps, markers):
        marker_mapping[np.float32(i)] = j
    return color_mapping, marker_mapping


def main():
    shape = (100, 100)
    n_true = 1000
    threshold = 2
    num_variants = 10
    n_trials = 5

    n_preds = np.linspace(1, 2000, num_variants, dtype=np.int32)
    props_tp = np.linspace(0.01, 1, num_variants, dtype=np.float32)

    results = []
    n_pred, prop_tp, precision, recall, f1score, mdr_recall, relative_recall = (
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    )

    for n_pred in track(n_preds):
        for prop_tp in props_tp:
            for _ in range(n_trials):
                gt_coords = get_random_coords(n_true, shape)
                pred_coords = simulate_a_model(
                    n_pred, prop_tp, shape, gt_coords, threshold
                )

                mdr_coords = get_random_coords(n_pred, shape)

                distances = cdist(pred_coords, gt_coords)
                mdr_distances = cdist(mdr_coords, gt_coords)

                precision, recall, f1score = (
                    assessment_utils.compute_precision_recall_f1score(
                        distances, threshold, num_pred=n_pred, num_true=n_true
                    )
                )
                mdr_precision, mdr_recall, mdr_f1score = (
                    assessment_utils.compute_precision_recall_f1score(
                        mdr_distances, threshold, num_pred=n_pred, num_true=n_true
                    )
                )
                relative_recall = assessment_utils.compute_relative_recall(
                    recall, mdr_recall
                )

                results.append(
                    [
                        n_pred,
                        prop_tp,
                        precision,
                        recall,
                        f1score,
                        mdr_recall,
                        1 - mdr_recall,
                        relative_recall,
                    ]
                )

    results = np.array(results)
    with open("evaluations/metric_validation/simulated_results.csv", "w") as outf:
        outf.write(
            "n_pred,prop_tp,precision,recall,f1score,mdr_recall,1 - mdr_recall,relative_recall\n"
        )
        for ln in results:
            outf.write(
                f"{ln[0]},{ln[1]},{ln[2]},{ln[3]},{ln[4]},{ln[5]},{ln[6]},{ln[7]}\n"
            )

    colors = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11"]
    markers = [".", "v", "^", "<", ">", "1", "2", "3", "4", "+", "x"]
    color_mapping, marker_mapping = get_color_and_marker_mappings(
        results, colors, markers
    )

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for ln in results:
        if ln[0] == "n_pred":
            continue
        # Relative recall vs F1-score
        ax[0].scatter(
            ln[4], ln[7], color=color_mapping[ln[0]], marker=marker_mapping[ln[1]]
        )
        ax[0].set_xlim(0, 1)
        ax[0].set_ylim(0, 1)
        ax[0].set_xlabel("F1-score")
        ax[0].set_ylabel("Relative recall")
        # 1-MDR recall vs Precision
        ax[1].scatter(
            ln[2], ln[6], color=color_mapping[ln[0]], marker=marker_mapping[ln[1]]
        )
        ax[1].set_xlim(0, 1)
        ax[1].set_ylim(0, 1)
        ax[1].set_xlabel("Precision")
        ax[1].set_ylabel("1 - MDR_recall")
    plt.tight_layout()
    plt.savefig(
        f"evaluations/metric_validation/{n_true}_gt_simulated_results_panelA.png",
        dpi=600,
    )
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for ln in results:
        if ln[0] == "n_pred":
            continue
        # F1-score vs num_predictions
        ax[0].scatter(ln[0], ln[4], color="Green", marker=marker_mapping[ln[1]])
        ax[0].set_ylim(0, 1)
        ax[0].set_xlabel("Number of predictions")
        ax[0].set_ylabel("F1-score")
        # Relative recall vs num_predictions
        ax[1].scatter(ln[0], ln[7], color="Green", marker=marker_mapping[ln[1]])
        ax[1].set_ylim(0, 1)
        ax[1].set_xlabel("Number of predictions")
        ax[1].set_ylabel("Relative recall")
    plt.tight_layout()
    plt.savefig(
        f"evaluations/metric_validation/{n_true}_gt_simulated_results_panelB.png",
        dpi=600,
    )
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for ln in results:
        if ln[0] == "n_pred":
            continue
        # Precision vs num_predictions
        ax[0].scatter(ln[0], ln[2], color="Green", marker=marker_mapping[ln[1]])
        ax[0].set_ylim(0, 1)
        ax[0].set_xlabel("Number of predictions")
        ax[0].set_ylabel("Precision")
        # 1-MDR_recall vs num_predictions
        ax[1].scatter(ln[0], ln[6], color="Green", marker=marker_mapping[ln[1]])
        ax[1].set_ylim(0, 1)
        ax[1].set_xlabel("Number of predictions")
        ax[1].set_ylabel("1 - MDR_recall")
    plt.tight_layout()
    plt.savefig(
        f"evaluations/metric_validation/{n_true}_gt_simulated_results_panelC.png",
        dpi=600,
    )
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for ln in results:
        if ln[0] == "n_pred":
            continue
        # Recall vs num_predictions
        ax[0].scatter(ln[0], ln[3], color="Green", marker=marker_mapping[ln[1]])
        ax[0].set_ylim(0, 1)
        ax[0].set_xlabel("Number of predictions")
        ax[0].set_ylabel("Recall")
        # MDR_recall vs num_predictions
        ax[1].scatter(ln[0], ln[5], color="Green", marker=marker_mapping[ln[1]])
        ax[1].set_ylim(0, 1)
        ax[1].set_xlabel("Number of predictions")
        ax[1].set_ylabel("MDR_recall")
    plt.tight_layout()
    plt.savefig(
        f"evaluations/metric_validation/{n_true}_gt_simulated_results_panelD.png",
        dpi=600,
    )
    plt.close()


if __name__ == "__main__":
    main()
