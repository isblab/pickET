import os
import sys
import time
import numpy as np
from rich.progress import track
from rich.console import Console
import socket
import slack_bot
from assets import utils, preprocessing, feature_extraction, clustering


def main():
    ### Input block
    console = Console()
    params_fname = sys.argv[1]
    params = utils.load_params_from_yaml(params_fname)

    dataset_name = params["dataset_name"]

    inputs = params["inputs"]
    output_dir = os.path.join(params["output_dir"], dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    window_size: int = params["window_size"]
    half_size: int = window_size // 2
    all_feature_extraction_params = params["feature_extraction_params"]
    max_num_windows_for_fitting = params.get("max_num_windows_for_fitting")
    clustering_methods = params["clustering_methods"]

    for feature_extraction_params in all_feature_extraction_params:
        feature_extractor = feature_extraction.FeatureExtractor(
            window_size, feature_extraction_params
        )

        ftex_mode = feature_extraction_params["mode"]
        for idx, target in enumerate(inputs):
            tic = time.perf_counter()
            z_lb = target.get("lower_z-slice_limit")
            z_ub = target.get("upper_z-slice_limit")
            console.print(
                f"Processing tomogram {idx+1}/{len(inputs)} using {ftex_mode} as the feature extraction method"
            )

            # Preprocess tomogram
            tomogram, voxel_sizes = preprocessing.load_tomogram(target["tomogram"])
            tomogram = preprocessing.guassian_blur_tomogram(tomogram)
            tomogram = preprocessing.minmax_normalize_array(tomogram)

            # Extract features
            with console.status(
                "Extracting all overlapping windows", spinner="bouncingBall"
            ) as status:
                windows, preshape = utils.get_windows(
                    preprocessing.get_z_section(tomogram, z_lb=z_lb, z_ub=z_ub),
                    window_size,
                    max_num_windows_for_fitting,
                )
                console.log("Successfully extracted all overalapping windows")
                status.update(status="Extracting features", spinner="bouncingBall")
                feature_extractor.extract_features(windows)
                console.log(
                    f"Feature extraction using method [cyan bold]{feature_extraction_params['mode']}[/] completed"
                )

            # Fitting the clusterers
            clusterers = {}
            for cl_method in clustering_methods:
                with console.status(
                    f"Clustering features using [cyan bold]{cl_method}[/]",
                    spinner="bouncingBall",
                ) as status:
                    if cl_method == "kmeans":
                        clusterer = clustering.fit_kmeans_clustering(
                            feature_extractor.features
                        )

                    elif cl_method == "gmm":
                        clusterer = clustering.fit_gmm_clustering(
                            feature_extractor.features
                        )
                    else:
                        raise ValueError(
                            f"Clustering method {cl_method} not supported.\
                                        Choose from 'kmeans', 'gmm'"
                        )
                    clusterers[cl_method]=clusterer

                    console.log(f"Fitting done for [cyan bold]{cl_method}[/] clusterer")
                    console.log(f"Preshape: {preshape}")
                    console.log(f"Features array of shape: {feature_extractor.features.shape}")

            # Build the segmentation
            segmentations = {}
            for cl_method,clusterer in clusterers.items():
                if (
                    (z_lb is not None)
                    or (z_ub is not None)
                    or (max_num_windows_for_fitting is not None)
                ):
                    segmentation = -1 * np.ones(tomogram.shape, dtype=np.int16)
                    for zslice_idx in track(
                        range(half_size, tomogram.shape[0] - half_size),
                        description="Building the segmentation",
                    ):
                        tomogram_zslab = tomogram[
                            zslice_idx - half_size : zslice_idx + half_size + 1
                        ]
                        slab_windows, slab_preshape = utils.get_windows(
                            tomogram_zslab, window_size
                        )
                        feature_extractor.extract_features(slab_windows)
                        slab_labels = clusterer.predict(feature_extractor.features)

                        seg = slab_labels.reshape(slab_preshape).astype(np.int16)
                        segmentation[
                            zslice_idx,
                            half_size : tomogram.shape[1] - half_size,
                            half_size : tomogram.shape[2] - half_size,
                        ] = seg

                else:
                    labels = clusterer.predict(feature_extractor.features)
                    segmentation = labels.reshape(preshape)
                    segmentation = np.pad(
                        segmentation, half_size, mode="constant", constant_values=-1
                    )

                console.log("Segmentation mask generated")
                segmentation = clustering.set_larger_cluster_as_bg(segmentation)
                segmentations[cl_method] = segmentation
                console.log(f"Segmentation using [cyan bold]{cl_method}[/] completed")

            toc = time.perf_counter()
            time_taken = toc - tic
            # Save the segmentation
            for cl_method, segmentation in segmentations.items():
                outf_full_path = os.path.join(
                    output_dir,
                    f"segmentation_{idx}_{ftex_mode}_{cl_method}.h5",
                )
                utils.save_segmentation(
                    outf_full_path,
                    segmentation=segmentation,
                    tomogram_path=target["tomogram"],
                    voxel_size=voxel_sizes,
                    window_size=window_size,
                    feature_extraction_params=feature_extraction_params,
                    clustering_method=cl_method,
                    max_num_windows_for_fitting=max_num_windows_for_fitting,
                    time_taken_for_s1=time_taken,
                )
                console.log(f"Segmentation saved to [cyan]{outf_full_path}[/]")
                
            console.print(
                f"Tomogram {idx+1} processed in {int(round(time_taken/60,0))} minutes\n"
            )

            message = f"Processed tomogram {idx+1}/{len(inputs)} "
            message += f"with {ftex_mode} as the feature extraction mode on {socket.gethostname()} "
            message += f"in {int(round(time_taken/60,0))} minutes"
            slack_bot.send_slack_dm(message)


if __name__ == "__main__":
    main()
