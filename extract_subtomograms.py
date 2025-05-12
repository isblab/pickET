import os
import sys
import yaml
import numpy as np
from assets import utils, particle_extraction


def main():
    coords_fpath = sys.argv[1]
    subtomogram_size = int(sys.argv[2])
    subtomogram_dir = sys.argv[3]

    with open(coords_fpath, "r") as coordsf:
        f_contents = yaml.safe_load(coordsf)
        metadata = f_contents["metadata"]
        coords = f_contents["PredictedCentroidCoordinates"]

    tomogram, _ = utils.load_tomogram(metadata["tomogram_path"])
    subtomograms = particle_extraction.extract_subtomograms(
        coords, subtomogram_size, tomogram
    )

    if not os.path.isdir(subtomogram_dir):
        os.mkdir(subtomogram_dir)

    for i, st in enumerate(subtomograms):
        np.save(os.path.join(subtomogram_dir, f"st_{i}.npy"), st)


if __name__ == "__main__":
    main()
