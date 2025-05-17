import os
import sys
import time
import datetime
import numpy as np
from rich.progress import track
from rich.console import Console
import socket
import slack_bot
from assets import utils, preprocessing, feature_extraction, clustering, segmentation_io


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

    neighborhood_size: int = params["neighborhood_size"]
    half_size: int = neighborhood_size // 2
    all_feature_extraction_params = params["feature_extraction_params"]
    clustering_methods = params["clustering_methods"]

    for feature_extraction_params in all_feature_extraction_params:
        feature_extractor = feature_extraction.FeatureExtractor(
            neighborhood_size, feature_extraction_params
        )

        ftex_mode = feature_extraction_params["mode"]
        for idx, target in enumerate(inputs):
            tic = time.perf_counter()
            z_lb = target.get("lower_z-slice_limit")
            z_ub = target.get("upper_z-slice_limit")
            max_num_neighborhoods_for_fitting = params.get(
                "max_num_neighborhoods_for_fitting"
            )
            console.print(
                f"Processing tomogram {idx+1}/{len(inputs)} using {ftex_mode} as the feature extraction method"
            )

            # Preprocess tomogram
            tomogram, voxel_sizes = utils.load_tomogram(target["tomogram"])
            tomogram = preprocessing.guassian_blur_tomogram(tomogram)
            tomogram = preprocessing.minmax_normalize_array(tomogram)

            # Extract features
            with console.status(
                "Extracting all overlapping neighborhoods", spinner="bouncingBall"
            ) as status:
                neighborhoods, preshape, max_num_neighborhoods_for_fitting = (
                    utils.get_neighborhoods(
                        preprocessing.get_z_section(tomogram, z_lb=z_lb, z_ub=z_ub),
                        neighborhood_size,
                        max_num_neighborhoods_for_fitting,
                    )
                )
                console.log("Successfully extracted all overalapping neighborhoods")
                console.log(f"Preshape: {preshape}")
                status.update(status="Extracting features", spinner="bouncingBall")
                feature_extractor.extract_features(neighborhoods)
                console.log(
                    f"Feature extraction using method [cyan bold]{feature_extraction_params['mode']}[/] completed"
                )
                console.log(
                    f"Features array of shape: {feature_extractor.features.shape}"
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
                    clusterers[cl_method] = clusterer

                    console.log(f"Fitting done for [cyan bold]{cl_method}[/] clusterer")

            # Build the segmentation
            segmentations = {}
            for cl_method, clusterer in clusterers.items():
                if (
                    (z_lb is not None)
                    or (z_ub is not None)
                    or (max_num_neighborhoods_for_fitting is not None)
                ):
                    segmentation = -1 * np.ones(tomogram.shape, dtype=np.int16)
                    for zslice_idx in track(
                        range(half_size, tomogram.shape[0] - half_size),
                        description="Building the segmentation",
                    ):
                        tomogram_zslab = tomogram[
                            zslice_idx - half_size : zslice_idx + half_size + 1
                        ]
                        slab_neighborhoods, slab_preshape, _ = utils.get_neighborhoods(
                            tomogram_zslab, neighborhood_size
                        )
                        feature_extractor.extract_features(slab_neighborhoods)
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
                segmentation_handler = segmentation_io.Segmentations()
                segmentation_handler.semantic_segmentation = segmentation

                segmentation_handler.metadata["dataset_name"] = params["dataset_name"]
                segmentation_handler.metadata["tomogram_path"] = target["tomogram"]

                vx_szs = [float(vx_sz) for vx_sz in voxel_sizes]
                segmentation_handler.metadata["voxel_size"] = vx_szs  # type:ignore
                segmentation_handler.metadata["neighborhood_size"] = (  # type:ignore
                    neighborhood_size
                )
                segmentation_handler.metadata["z_lb_for_clusterer_fitting"] = z_lb
                segmentation_handler.metadata["z_ub_for_clusterer_fitting"] = z_ub
                segmentation_handler.metadata["max_num_neighborhoods_for_fitting"] = (
                    max_num_neighborhoods_for_fitting
                )
                segmentation_handler.metadata["fex_mode"] = ftex_mode  # type:ignore
                if ftex_mode == "ffts":
                    segmentation_handler.metadata["fex_n_fft_subsets"] = (
                        feature_extraction_params["n_fft_subsets"]
                    )

                elif ftex_mode == "gabor":
                    segmentation_handler.metadata["fex_num_neighborhoods_subsets"] = (
                        feature_extraction_params["num_neighborhoods_subsets"]
                    )
                    segmentation_handler.metadata["fex_num_sinusoids"] = (
                        feature_extraction_params["num_sinusoids"]
                    )
                    segmentation_handler.metadata["fex_num_parallel_filters"] = (
                        feature_extraction_params["num_parallel_filters"]
                    )
                    segmentation_handler.metadata["fex_num_output_features"] = (
                        feature_extraction_params["num_output_features"]
                    )

                segmentation_handler.metadata["clustering_method"] = cl_method
                segmentation_handler.metadata["time_taken_for_s1"] = (  # type:ignore
                    time_taken
                )
                segmentation_handler.metadata["timestamp_s1"] = (  # type:ignore
                    datetime.datetime.now().isoformat()
                )

                segmentation_handler.generate_output_file(outf_full_path)
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
