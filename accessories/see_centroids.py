import sys
import yaml
import napari
import numpy as np
from assets import utils


def main():
    pred_centroids_fname = sys.argv[1]

    spot_size = 4
    half_spot_size = spot_size // 2

    with open(pred_centroids_fname, "r") as predcf:
        contents = yaml.safe_load(predcf)
        metadata = contents["metadata"]
        read_coords = contents["Predicted_Particle_Centroid_Coordinates"]

    tomogram, _ = utils.load_tomogram(metadata["tomogram_path"])

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

    viewer = napari.Viewer()
    viewer.add_image(tomogram, name="Tomogram")
    viewer.add_labels(mask, name="Segmentation")
    napari.run()


if __name__ == "__main__":
    main()
