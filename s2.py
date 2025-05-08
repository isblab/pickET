import os
import sys
import time
import numpy as np

import slack_bot
from assets import utils, particle_extraction


def main():
    ### Input block
    params_fname = sys.argv[1]

    params = utils.load_params_from_yaml(params_fname)

    dataset_name = params["dataset_name"]
    inputs = params["inputs"]
    particle_extraction_params = params["particle_extraction_params"]
    output_dir = os.path.join(params["output_dir"], dataset_name, "predicted_particles")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    ### Processing block
    for idx, target in enumerate(inputs):
        tic = time.perf_counter()
        print(f"Processing segmentation {idx+1}/{len(inputs)}")

        particle_cluster_id = target["particle_cluster_id"]
        segmentation, h5_metadata = utils.load_h5file(target["segmentation"])

        clustering_method = h5_metadata["clustering_method"]

        segmentation = np.where(segmentation == particle_cluster_id, 1, 0)

        for p_ex_params in particle_extraction_params:
            instance_seg, num_objects = particle_extraction.do_instance_segmentation(
                segmentation, p_ex_params
            )
            centroid_coords = particle_extraction.get_centroids(
                instance_seg, num_objects
            )
            utils.write_coords_as_ndjson(
                centroid_coords,
                os.path.join(
                    output_dir,
                    f"predicted_centroids_{idx}_{clustering_method}_{p_ex_params['mode']}.ndjson",
                ),
            )
        toc = time.perf_counter()
        time_taken = toc - tic
        print(f"\tTomogram {idx+1} processed in {time_taken:.2f} seconds\n")

    slack_bot.send_slack_dm(
        f"The python process with parameter file name: '{params_fname}' completed"
    )


if __name__ == "__main__":
    main()
