import numpy as np
from assets import utils

fname = "/data2/shreyas/mining_tomograms/working/s1_clean_results_picket/czi_ds_10440_denoised/intensities/predicted_particles/predicted_centroids_0_kmeans_connected_component_labeling.ndjson"
shape = (184, 630, 630)
coords = utils.read_ndjson_coords(fname)

mask = np.zeros(shape)
for c in coords:
    mask[
        int(c[0] - 3) : int(c[0] + 3 + 1),
        int(c[1] - 3) : int(c[1] + 3 + 1),
        int(c[2] - 3) : int(c[2] + 3 + 1),
    ] = 1

np.save("pred_coords_mask.npy", mask)
