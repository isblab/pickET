import sys
import yaml
import numpy as np
from picket.core import utils


def main():
    pred_centroids_fname = sys.argv[1]

    spot_size = 10
    half_spot_size = spot_size // 2

    with open(pred_centroids_fname, "r") as predcf:
        contents = yaml.safe_load(predcf)
        metadata = contents["metadata"]
        read_coords = contents["Predicted_Particle_Centroid_Coordinates"]

    tomo_path = metadata["tomogram_path"]
    #! Deprecated: Remove this section later on
    # tomo_path2 = tomo_path.split("/")[2:]
    # tomo_path = os.path.join("/data2", *tomo_path2)

    tomogram, _ = utils.load_tomogram(tomo_path)

    coords = np.nan * np.ones((len(read_coords), 3))
    for idx, ln in enumerate(read_coords):
        z, y, x = ln["z"], ln["y"], ln["x"]
        coords[idx] = np.array([z, y, x])
    coords = coords.astype(np.int32)

    mask = np.zeros(tomogram.shape, dtype=np.int16)
    for c in coords:
        mask[
            c[0] - half_spot_size : c[0] + half_spot_size + 1,
            c[1] - half_spot_size : c[1] + half_spot_size + 1,
            c[2] - half_spot_size : c[2] + half_spot_size + 1,
        ] = 1

    utils.load_in_napari(tomogram, mask, segname="Predicted particle centroids")


if __name__ == "__main__":
    main()
