import os
import sys
import time
import numpy as np
import scipy.ndimage as nd

from assets import utils, particle_extraction


def main():
    ### Input block
    params_fname = sys.argv[1]
    out_fname_suffix = sys.argv[2]

    params = utils.load_params_from_yaml(params_fname)

    experiment_name = params["experiment_name"]
    run_name = params["run_name"]
    inputs = params["inputs"]
    output_dir = os.path.join(params["output_dir"], experiment_name, run_name)
    particle_extraction_params = params["particle_extraction_params"]

    ### Processing block
    for idx, target in enumerate(inputs):
        tic = time.perf_counter()
        print(f"Processing segmentation {idx+1}/{len(inputs)}")

        segmentation = np.load(target["segmentation"])

        for p_ex_params in particle_extraction_params:
            instance_seg, num_objects = particle_extraction.do_instance_segmentation(
                segmentation, p_ex_params
            )
            centroid_coords = particle_extraction.get_centroids(
                instance_seg, num_objects
            )
            particle_extraction.write_coords_as_ndjson(
                centroid_coords,
                os.path.join(
                    output_dir, f"predicted_centroids_{out_fname_suffix}.ndjson"
                ),
            )


if __name__ == "__main__":
    main()
