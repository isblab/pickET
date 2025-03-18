import os
import sys
import time
import numpy as np
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
        segmentation = np.zeros(tomogram.shape, dtype=np.uint8)

        tomogram = preprocessing.get_z_section(
            tomogram,
            z_lb=target.get("lower_z-slice_limit"),
            z_ub=target.get("upper_z-slice_limit"),
        )
        tomogram = preprocessing.guassian_blur_tomogram(tomogram)
        tomogram = preprocessing.minmax_normalize_array(tomogram)

        # Extract features
        feature_extractor.get_windows(tomogram)
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
            seg = np.pad(seg, window_size // 2, mode="constant", constant_values=0)
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


if __name__ == "__main__":
    main()
