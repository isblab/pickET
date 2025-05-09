import os
import sys
import time
import yaml
import numpy as np

import slack_bot
from assets import utils, preprocessing, particle_extraction


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
        z_lb = target.get("lower_z-slice_limit")
        z_ub = target.get("upper_z-slice_limit")

        # Load segmentation
        seg_path = target["segmentation"]
        segmentation, h5_metadata = utils.load_h5file(seg_path)
        h5_metadata["particle_cluster_id"] = particle_cluster_id
        h5_metadata["segmentation_path"] = seg_path

        segmentation = np.where(segmentation == particle_cluster_id, 1, 0)
        if z_lb is not None:
            segmentation[:z_lb] = 0
        if z_ub is not None:
            segmentation[z_ub + 1 :] = 0

        outputs = {}
        for p_ex_params in particle_extraction_params:
            instance_seg, num_objects = particle_extraction.do_instance_segmentation(
                segmentation, p_ex_params
            )
            centroid_coords = particle_extraction.get_centroids(
                instance_seg, num_objects
            )
            out_fpath = os.path.join(
                output_dir,
                f"predicted_centroids_{idx}_{h5_metadata["clustering_method"]}_{p_ex_params['mode']}.yaml",
            )
            outputs[out_fpath] = utils.prepare_out_coords(
                coords=centroid_coords,
                metadata=h5_metadata,
                z_lb=z_lb,
                z_ub=z_ub,
            )
        toc = time.perf_counter()
        time_taken = toc - tic

        for out_fname, out_dict in outputs.items():
            out_dict["metadata"]["time_taken_for_s2"] = time_taken
            with open(out_fname, "w") as out_annot_f:
                yaml.dump(out_dict, out_annot_f, sort_keys=False)

        print(f"\tTomogram {idx+1} processed in {time_taken:.2f} seconds\n")

    slack_bot.send_slack_dm(
        f"The python process with parameter file name: '{params_fname}' completed"
    )


if __name__ == "__main__":
    main()
