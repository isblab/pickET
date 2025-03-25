import sys
import mrcfile
import numpy as np
from assets import utils

tomo_path = sys.argv[1]
annot_path = sys.argv[2]
pred_path = sys.argv[3]

tomo = mrcfile.read(tomo_path)

t_centroids = utils.read_ndjson_coords(annot_path)
p_centroids = utils.read_ndjson_coords(pred_path)

t_mask, p_mask = np.zeros(tomo.shape), np.zeros(tomo.shape)
for c in t_centroids:
    t_mask[
        int(c[0] - 3) : int(c[0] + 3),
        int(c[1] - 3) : int(c[1] + 3),
        int(c[2] - 3) : int(c[2] + 3),
    ] = 2

for c in p_centroids:
    p_mask[
        int(c[0] - 3) : int(c[0] + 3),
        int(c[1] - 3) : int(c[1] + 3),
        int(c[2] - 3) : int(c[2] + 3),
    ] = 1


print(f"Ground truth: {len(t_centroids)}\nPredicted: {len(p_centroids)}")

mrcfile.new(
    "/home/shreyas/Projects/mining_tomograms/github/pickET/centroid_visualization/ground_truth_labels.mrc",
    t_mask.astype(np.int16),
    overwrite=True,
)

mrcfile.new(
    "/home/shreyas/Projects/mining_tomograms/github/pickET/centroid_visualization/predicted_labels.mrc",
    p_mask.astype(np.int16),
    overwrite=True,
)

mrcfile.new(
    "/home/shreyas/Projects/mining_tomograms/github/pickET/centroid_visualization/occupancy.mrc",
    tomo.astype(np.float32),
    overwrite=True,
)
