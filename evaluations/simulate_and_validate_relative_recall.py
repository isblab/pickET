import os
import sys
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
    out_dir = sys.argv[1]
    shape = (100, 100)
    n_true = 500
    threshold = 2
    pred_count = 10
    num_tp_props = 5
    n_trials = 5

    n_preds = np.linspace(100, 2000, pred_count, dtype=np.int32)
    props_tp = np.linspace(0.2, 0.8, num_tp_props, dtype=np.float32)

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
    with open(os.path.join(out_dir, "simulated_results.csv"), "w") as outf:
        outf.write(
            "n_pred,prop_tp,precision,recall,f1score,mdr_recall,1 - mdr_recall,relative_recall\n"
        )
        for ln in results:
            outf.write(
                f"{ln[0]},{ln[1]},{ln[2]},{ln[3]},{ln[4]},{ln[5]},{ln[6]},{ln[7]}\n"
            )

    colors = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11"]
    markers = ["o", "^", "s", "*", "P", "D"]
    color_mapping, marker_mapping = get_color_and_marker_mappings(
        results, colors, markers
    )

    markers_used = []
    fig = plt.figure(figsize=(8, 8))
    for ln in results:
        if ln[0] == "n_pred":
            continue
        # F1-score vs num_predictions and relative recall vs num_predictions
        m = marker_mapping[ln[1]]
        if m in markers_used:
            plt.scatter(ln[0], ln[4], color="Green", marker=m)

        else:
            label = f"Precision = {float(ln[1]):.2f}"
            plt.scatter(ln[0], ln[4], color="Green", marker=m, label=label)

            markers_used.append(m)

    plt.ylim(0, 1)
    plt.xlabel("Number of predictions")
    plt.ylabel("F1-score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"{n_true}_gt_simulated_results_panelC.png"), dpi=600
    )
    plt.close()

    markers_used = []
    fig = plt.figure(figsize=(8, 8))
    for ln in results:
        if ln[0] == "n_pred":
            continue
        # F1-score vs num_predictions and relative recall vs num_predictions
        m = marker_mapping[ln[1]]
        if m in markers_used:
            plt.scatter(ln[0], ln[7], color="Green", marker=m)
        else:
            label = f"Precision = {float(ln[1]):.2f}"
            plt.scatter(ln[0], ln[7], color="Green", marker=m, label=label)
            markers_used.append(m)

    plt.ylim(0, 1)
    plt.xlabel("Number of predictions")
    plt.ylabel("Relative recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"{n_true}_gt_simulated_results_panelD.png"), dpi=600
    )
    plt.close()

    markers_used = []
    fig = plt.figure(figsize=(8, 8))
    for ln in results:
        if ln[0] == "n_pred":
            continue
        # Precision vs num_predictions and complement of random recall vs num_predictions
        m = marker_mapping[ln[1]]
        if m in markers_used:
            plt.scatter(ln[0], ln[6], color="Green", marker=m)
        else:
            label = f"Precision = {float(ln[1]):.2f}"
            plt.scatter(ln[0], ln[6], color="Green", marker=m, label=label)
            markers_used.append(m)

    plt.ylim(0, 1)
    plt.xlabel("Number of predictions")
    plt.ylabel("Complement of random recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"{n_true}_gt_simulated_results_panelB.png"), dpi=600
    )
    plt.close()

    markers_used = []
    fig = plt.figure(figsize=(8, 8))
    for ln in results:
        if ln[0] == "n_pred":
            continue
        # Recall vs num_predictions and MDR_recall vs num_predictions
        m = marker_mapping[ln[1]]
        if m in markers_used:
            plt.scatter(ln[0], ln[3], color="Green", marker=m)

        else:
            label = f"Precision = {float(ln[1]):.2f}"
            plt.scatter(ln[0], ln[3], color="Green", marker=m, label=label)
            markers_used.append(m)

    plt.ylim(0, 1)
    plt.xlabel("Number of predictions")
    plt.ylabel("Recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"{n_true}_gt_simulated_results_panelA.png"), dpi=600
    )
    plt.close()


if __name__ == "__main__":
    main()
