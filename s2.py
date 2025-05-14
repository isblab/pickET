import os
import sys
import yaml
import time
import datetime
import numpy as np

import slack_bot
from assets import utils, preprocessing, particle_extraction, segmentation_io


def main():
    ### Input block
    params_fname = sys.argv[1]

    params = utils.load_params_from_yaml(params_fname)

    dataset_name = params["dataset_name"]
    inputs = params["inputs"]
    particle_extraction_params = params["particle_extraction_params"]

    iseg_output_dir = os.path.join(
        params["output_dir"], dataset_name, "instance_segmentations"
    )
    coords_output_dir = os.path.join(
        params["output_dir"], dataset_name, "predicted_particles"
    )
    if not os.path.isdir(coords_output_dir):
        os.mkdir(coords_output_dir)
    if not os.path.isdir(iseg_output_dir):
        os.mkdir(iseg_output_dir)

    ### Processing block
    for idx, target in enumerate(inputs):
        time_taken_per_tomo = 0
        tic = time.perf_counter()
        print(f"Processing segmentation {idx+1}/{len(inputs)}")

        particle_cluster_id = target["particle_cluster_id"]
        z_lb = target.get("lower_z-slice_limit")
        z_ub = target.get("upper_z-slice_limit")

        # Load segmentation
        seg_path = target["segmentation"]
        segmentation_handler = segmentation_io.Segmentations()
        segmentation_handler.load_segmentations(seg_path)
        semantic_segmentation = segmentation_handler.semantic_segmentation

        segmentation = np.where(semantic_segmentation == particle_cluster_id, 1, 0)
        if z_lb is not None:
            segmentation[:z_lb] = 0
        if z_ub is not None:
            segmentation[z_ub + 1 :] = 0

        segmentation_handler.metadata["z_lb_for_particle_extraction"] = z_lb
        segmentation_handler.metadata["z_ub_for_particle_extraction"] = z_ub
        segmentation_handler.metadata["particle_cluster_id"] = particle_cluster_id

        coords_outputs = {}
        for p_ex_params in particle_extraction_params:
            instance_seg, num_objects = particle_extraction.do_instance_segmentation(
                segmentation, p_ex_params
            )
            centroid_coords = particle_extraction.get_centroids(
                instance_seg, num_objects
            )

            segmentation_handler.instance_segmentation = instance_seg
            segmentation_handler.metadata["pex_mode"] = p_ex_params["mode"]
            if p_ex_params["mode"] == "watershed_segmentation":
                segmentation_handler.metadata["pex_min_distance"] = p_ex_params[
                    "min_distance"
                ]

            iseg_out_fpath = os.path.join(
                iseg_output_dir,
                f"instance_segmentation_{idx}_{segmentation_handler.metadata["fex_mode"]}_{segmentation_handler.metadata["clustering_method"]}_{p_ex_params['mode']}.yaml",
            )
            coords_out_fpath = os.path.join(
                coords_output_dir,
                f"predicted_centroids_{idx}_{segmentation_handler.metadata["fex_mode"]}_{segmentation_handler.metadata["clustering_method"]}_{p_ex_params['mode']}.yaml",
            )

            toc = time.perf_counter()
            time_taken = toc - tic
            time_taken_per_tomo += time_taken

            segmentation_handler.metadata["time_taken_for_s2"] = (  # type:ignore
                time_taken
            )
            segmentation_handler.metadata["timestamp_s2"] = (  # type:ignore
                datetime.datetime.now().isoformat()
            )

            coords_outputs = utils.prepare_out_coords(
                coords=centroid_coords,
                metadata=segmentation_handler.metadata,
            )

            with open(coords_out_fpath, "w") as out_annot_f:
                yaml.dump(coords_outputs, out_annot_f, sort_keys=True)
            segmentation_handler.generate_output_file(iseg_out_fpath)

        print(f"\tTomogram {idx+1} processed in {time_taken_per_tomo:.2f} seconds\n")

    slack_bot.send_slack_dm(
        f"The python process with parameter file name: '{params_fname}' completed"
    )


if __name__ == "__main__":
    main()
