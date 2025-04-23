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
    half_size = window_size // 2
    all_feature_extraction_params = params["feature_extraction_params"]
    max_num_windows_for_fitting = params.get("max_num_windows_for_fitting")
    clustering_methods = params["clustering_methods"]

    for feature_extraction_params in all_feature_extraction_params:
        feature_extractor = feature_extraction.FeatureExtractor(
            window_size, feature_extraction_params
        )
        ftex_mode = feature_extraction_params["mode"]

        ### Processing block
        for idx, target in enumerate(inputs):
            tic = time.perf_counter()
            z_lb = target.get("lower_z-slice_limit")
            z_ub = target.get("upper_z-slice_limit")
            console.print(
                f"Processing tomogram {idx+1}/{len(inputs)} using {ftex_mode} as the feature extraction method"
            )

            # Preprocess tomogram
            tomogram = preprocessing.load_tomogram(target["tomogram"])

            tomogram_init_sec = preprocessing.get_z_section(
                tomogram, z_lb=z_lb, z_ub=z_ub
            )
            tomogram_init_sec = preprocessing.guassian_blur_tomogram(tomogram_init_sec)
            tomogram_init_sec = preprocessing.minmax_normalize_array(tomogram_init_sec)

            # Extract features
            with console.status(
                "Extracting all overlapping windows", spinner="bouncingBall"
            ) as status:
                windows, preshape = utils.get_windows(
                    tomogram_init_sec, window_size, max_num_windows_for_fitting
                )
                console.log("Successfully extracted all overalapping windows")
                status.update(status="Extracting features", spinner="bouncingBall")
                feature_extractor.extract_features(windows)
                console.log(
                    f"Feature extraction using method [cyan bold]{feature_extraction_params['mode']}[/] completed"
                )
                np.save(
                    os.path.join(output_dir, f"old_features.npy"),
                    feature_extractor.features,
                )

            # Cluster features
            for cl_method in clustering_methods:
                with console.status(
                    f"Clustering features using [cyan bold]{cl_method}[/]",
                    spinner="bouncingBall",
                ) as status:
                    if cl_method == "kmeans":
                        clusterer = clustering.fit_kmeans_clustering(
                            feature_extractor.features
                        )
                        l1 = clusterer.labels_.reshape(
                            preshape[0], preshape[1], preshape[2]
                        )
                        l1 = np.pad(l1, half_size, mode="constant", constant_values=0)
                        np.save(
                            os.path.join(
                                output_dir,
                                f"old_segmentation_.npy",
                            ),
                            l1,
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

                    console.log(f"Fitting done for [cyan bold]{cl_method}[/] clusterer")

                new_ft = []
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
                        new_ft.append(feature_extractor.features)
                        slab_labels = clusterer.predict(feature_extractor.features)

                        seg = slab_labels.reshape(slab_preshape).astype(np.int16)
                        segmentation[
                            zslice_idx,
                            half_size : tomogram.shape[1] - half_size,
                            half_size : tomogram.shape[2] - half_size,
                        ] = seg

                    new_ft = np.concatenate(new_ft, axis=0)
                    np.save(os.path.join(output_dir, f"new_features.npy"), new_ft)
                else:
                    labels = clusterer.predict(feature_extractor.features)
                    segmentation = labels.reshape(preshape)
                    segmentation = np.pad(
                        segmentation, half_size, mode="constant", constant_values=-1
                    )

                console.log("Segmentation mask generated")
                segmentation = clustering.set_larger_cluster_as_bg(segmentation)

                outf_full_path = os.path.join(
                    output_dir,
                    f"segmentation_{idx}_{ftex_mode}_{cl_method}.npy",
                )
                np.save(
                    outf_full_path,
                    segmentation.astype(np.int16),
                )
                console.log(f"Segmentation saved to [cyan]{outf_full_path}[/]")

            toc = time.perf_counter()
            time_taken = toc - tic
            console.print(
                f"Tomogram {idx+1} processed in {int(round(time_taken/60,0))} minutes\n"
            )

            message = f"Processed tomogram {idx+1}/{len(inputs)} "
            message += f"with {ftex_mode} as the feature extraction mode on {socket.gethostname()} "
            message += f"in {int(round(time_taken/60,0))} minutes"
            slack_bot.send_slack_dm(message)


if __name__ == "__main__":
    main()
