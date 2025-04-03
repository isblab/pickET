import os
import sys
import time
import numpy as np

import socket
import slack_bot
from assets import utils, preprocessing, feature_extraction, clustering


def main():
    ### Input block
    params_fname = sys.argv[1]
    params = utils.load_params_from_yaml(params_fname)

    experiment_name = params["experiment_name"]
    run_name = params["run_name"]
    inputs = params["inputs"]
    output_dir = os.path.join(params["output_dir"], experiment_name, run_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    window_size = params["window_size"]
    feature_extraction_params = params["feature_extraction_params"]
    clustering_methods = params["clustering_methods"]

    feature_extractor = feature_extraction.FeatureExtractor(
        window_size, feature_extraction_params
    )

    ### Processing block
    for idx, target in enumerate(inputs):
        tic = time.perf_counter()
        print(f"Processing tomogram {idx+1}/{len(inputs)}")

        # Preprocess tomogram
        tomogram = preprocessing.load_tomogram(target["tomogram"])
        segmentation = -1 * np.ones(tomogram.shape, dtype=np.int16)

        tomogram = preprocessing.get_z_section(
            tomogram,
            z_lb=target.get("lower_z-slice_limit"),
            z_ub=target.get("upper_z-slice_limit"),
        )
        tomogram = preprocessing.guassian_blur_tomogram(tomogram)
        tomogram = preprocessing.minmax_normalize_array(tomogram)

        # Extract features
        feature_extractor.get_windows(tomogram)

        is_memory_available = (False, -1, -1)
        if (
            feature_extraction_params["mode"] == "ffts"
            or feature_extraction_params["mode"] == "intensities"
        ):
            is_memory_available = utils.check_memory_availability(
                feature_extractor.windows.shape[0],
                window_size**3,
            )
        elif feature_extraction_params["mode"] == "gabor":
            is_memory_available = utils.check_memory_availability(
                feature_extractor.windows.shape[0],
                feature_extraction_params["num_output_features"] * 2,
            )

        print(
            f"\tIs memory available: {is_memory_available[0]}\tAvailable: {is_memory_available[1]}GB\tRequired: {is_memory_available[2]}GB"
        )
        if not is_memory_available[0]:
            slack_bot.send_slack_dm(
                f"Error: The python process with parameter file name: '{params_fname}' terminated due to MemoryError on {socket.gethostname()}"
            )
            raise MemoryError("Not enough memory available")

        print("\tExtracting features")
        feature_extractor.extract_features()

        # Cluster features
        for cl_method in clustering_methods:
            if cl_method == "kmeans":
                labels = clustering.do_kmeans_clustering(feature_extractor.features)
            elif cl_method == "gmm":
                labels, _ = clustering.do_gmm_clustering(feature_extractor.features)
            else:
                raise ValueError(
                    f"Clustering method {cl_method} not supported.\
                                 Choose from 'kmeans', 'gmm'"
                )

            seg = labels.reshape(feature_extractor.preshape)
            seg = np.pad(seg, window_size // 2, mode="constant", constant_values=-1)
            segmentation[
                target.get("lower_z-slice_limit") : target.get("upper_z-slice_limit")
            ] = seg

            np.save(
                os.path.join(output_dir, f"segmentation_{idx}_{cl_method}.npy"),
                segmentation,
            )
        toc = time.perf_counter()
        time_taken = toc - tic
        print(
            f"\tTomogram {idx+1} processed in {int(time_taken//60)}:{time_taken%60:.0f} minutes\n"
        )

    slack_bot.send_slack_dm(
        f"The python process with parameter file name: '{params_fname}' completed on {socket.gethostname()}"
    )


if __name__ == "__main__":
    main()
