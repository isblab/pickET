import os
import sys
import time
import numpy as np

sys.path.append("/home/shreyas/Projects/accessory_scripts/")
import slack_bot
from assets import utils, particle_extraction


def main():
    ### Input block
    params_fname = sys.argv[1]

    params = utils.load_params_from_yaml(params_fname)

    experiment_name = params["experiment_name"]
    run_name = params["run_name"]
    inputs = params["inputs"]
    clustering_methods = params["clustering_method"]
    particle_extraction_params = params["particle_extraction_params"]
    output_dir = os.path.join(
        params["output_dir"], experiment_name, run_name, "predicted_particles"
    )
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

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
            utils.write_coords_as_ndjson(
                centroid_coords,
                os.path.join(
                    output_dir,
                    f"predicted_centroids_{idx}_{clustering_methods}_{p_ex_params['mode']}.ndjson",
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
