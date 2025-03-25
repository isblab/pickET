import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_distance(a_coords, b_coords):
    d_square = (
        (a_coords[0] - b_coords[0]) ** 2
        + (a_coords[1] - b_coords[1]) ** 2
        + (a_coords[2] - b_coords[2]) ** 2
    )
    return np.sqrt(d_square)


threshold = 0.6
box_size = 29
overall_size = 100
coord_true = [50, 50, 50]

ov_true = torch.tensor(
    np.zeros((overall_size, overall_size, overall_size)), device="cuda"
)
ov_true[
    coord_true[0] - box_size // 2 : coord_true[0] + box_size // 2 + 1,
    coord_true[1] - box_size // 2 : coord_true[1] + box_size // 2 + 1,
    coord_true[2] - box_size // 2 : coord_true[2] + box_size // 2 + 1,
] = 1


positive_distances = []
iou_mask = np.zeros((overall_size, overall_size, overall_size))
for zi in tqdm(range(box_size // 2, overall_size - box_size // 2)):
    for yi in range(box_size // 2, overall_size - box_size // 2):
        for xi in range(box_size // 2, overall_size - box_size // 2):
            coord_pred = [zi, yi, xi]
            ov_pred = torch.tensor(
                np.zeros((overall_size, overall_size, overall_size)), device="cuda"
            )
            ov_pred[
                coord_pred[0] - box_size // 2 : coord_pred[0] + box_size // 2 + 1,
                coord_pred[1] - box_size // 2 : coord_pred[1] + box_size // 2 + 1,
                coord_pred[2] - box_size // 2 : coord_pred[2] + box_size // 2 + 1,
            ] = 1

            overall_arr = ov_true + ov_pred

            intersection = torch.sum(torch.where(overall_arr == 2, 1, 0))
            union = torch.count_nonzero(overall_arr)
            iou = float(intersection / union)
            if iou >= threshold:
                iou_mask[zi, yi, xi] = iou
                positive_distances.append(get_distance(coord_true, coord_pred))

np.save("iou_mask.npy", iou_mask)
plt.hist(positive_distances)
plt.xlabel(f"Distances with IoU >= {threshold}")
plt.savefig(
    f"/home/shreyas/Projects/mining_tomograms/github/pickET/centroid_distances_io>={threshold}_bs={box_size}.png"
)
