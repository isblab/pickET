import os
import sys
import ndjson
import numpy as np
from tqdm import tqdm

from assets import utils
import assessment_utils


def read_annotations_ndjson_file(fname: str) -> dict[str, np.ndarray]:
    with open(fname, "r") as in_annot_f:
        annotations = ndjson.load(in_annot_f)

    p_deets = {}
    for ln in annotations:
        particle_id = ln["particle_id"]
        if particle_id not in p_deets:
            p_deets[particle_id] = [
                np.array(
                    [ln["location"]["z"], ln["location"]["y"], ln["location"]["x"]]
                )
            ]
        else:
            p_deets[particle_id].append(
                np.array(
                    [ln["location"]["z"], ln["location"]["y"], ln["location"]["x"]]
                )
            )
    return p_deets


def main():
    ### Input block
    params_fname = sys.argv[1]

    params = utils.load_params_from_yaml(params_fname)

    experiment_name = params["experiment_name"]
    run_name = params["run_name"]
    angstrom_threshold = params["threshold_in_angstrom"]
    inputs = params["inputs"]
    clustering_method = params["clustering_method"]
    particle_extraction_method = params["particle_extraction_method"]
    output_dir = os.path.join(params["output_dir"], experiment_name, run_name)

    results = {}
    ### Processing block
    for idx, target in enumerate(tqdm(inputs)):
        # print(f"Processing prediction {idx+1}/{len(inputs)}")
        threshold = assessment_utils.get_voxel_threshold(
            target["tomogram"], angs_threshold=angstrom_threshold
        )
        pred_centroids = utils.read_ndjson_coords(target["predicted_centroids"])
        true_centroids = read_annotations_ndjson_file(target["true_centroids"])
        print(true_centroids)
        exit()


if __name__ == "__main__":
    main()
