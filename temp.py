import numpy as np

box_size = 37

coord_true = [25, 25, 25]
coord_pred = [30, 24, 28]

distances = np.sqrt(
    (coord_pred[0] - coord_true[0]) ** 2
    + (coord_pred[1] - coord_true[1]) ** 2
    + (coord_pred[2] - coord_true[2]) ** 2
)
print(f"Distances: {distances:.4f}")

ov_true = np.zeros((100, 100, 100))
ov_true[
    coord_true[0] - box_size // 2 : coord_true[0] + box_size // 2 + 1,
    coord_true[1] - box_size // 2 : coord_true[1] + box_size // 2 + 1,
    coord_true[2] - box_size // 2 : coord_true[2] + box_size // 2 + 1,
] = 1

ov_pred = np.zeros((100, 100, 100))
ov_pred[
    coord_pred[0] - box_size // 2 : coord_pred[0] + box_size // 2 + 1,
    coord_pred[1] - box_size // 2 : coord_pred[1] + box_size // 2 + 1,
    coord_pred[2] - box_size // 2 : coord_pred[2] + box_size // 2 + 1,
] = 1

overall_arr = ov_true + ov_pred

intersection = np.sum(np.where(overall_arr == 2, 1, 0))
union = np.count_nonzero(overall_arr)
print(
    f"Intersection: {intersection}\nUnion: {union}\nIoU: {(intersection / union):.4f}"
)
