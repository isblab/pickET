import os
import sys
import time
import numpy as np
from tqdm import tqdm
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

        print("Getting baseline")
        feature_extractor.get_windows(tomogram)
        print(f"{feature_extractor.windows.shape}")

        for z_idx in tqdm(
            range(window_size // 2, tomogram.shape[0] - (window_size // 2))
        ):
            sub_z = tomogram[z_idx - window_size // 2 : z_idx + (window_size // 2) + 1]

            feature_extractor.get_windows(sub_z)
            print(sub_z.shape, feature_extractor.windows.shape)

        # # Extract features
        # feature_extractor.get_windows(tomogram)
        # feature_extractor.extract_features()

        exit()


if __name__ == "__main__":
    main()
