import numpy as np
from rich.progress import track
from icecream import ic
from assets import utils, preprocessing, feature_extraction, clustering


def main():
    ### Input block
    tomo_path = "/data2/shreyas/mining_tomograms/datasets/tomotwin/tomo_simulation_round_1/tomo_01.2022-04-11T140327+0200/denoised_tiltseries_rec.mrc"
    window_size: int = 5
    half_size = window_size // 2
    feature_extraction_params = {
        "mode": "ffts",
        "n_fft_subsets": 64,
        "use_subset_size": 10_000_000,
        "num_output_features": 64,
    }

    feature_extractor = feature_extraction.FeatureExtractor(
        window_size, feature_extraction_params
    )

    tomogram = preprocessing.load_tomogram(tomo_path)
    tomogram = preprocessing.guassian_blur_tomogram(tomogram)
    tomogram = preprocessing.minmax_normalize_array(tomogram)

    windows, preshape = utils.get_windows(tomogram, window_size)

    feature_extractor.extract_features(windows)
    old_features = feature_extractor.features
    ic(old_features.shape)
    np.save(
        "/data2/shreyas/mining_tomograms/working/tests/first_pass_features.npy",
        old_features,
    )
    clusterer = clustering.fit_kmeans_clustering(old_features)
    l1 = clusterer.labels_.reshape(preshape[0], preshape[1], preshape[2])

    sp_features = []
    l2 = []
    for zslice_idx in track(
        range(half_size, tomogram.shape[0] - half_size),
        description="Building the segmentation",
    ):
        tomogram_zslab = tomogram[zslice_idx - half_size : zslice_idx + half_size + 1]
        slab_windows, slab_preshape = utils.get_windows(tomogram_zslab, window_size)
        feature_extractor.extract_features(slab_windows)
        sp_features.append(
            feature_extractor.features.reshape(
                (
                    slab_preshape[0],
                    slab_preshape[1],
                    slab_preshape[2],
                    feature_extraction_params["num_output_features"],
                )
            )
        )
        l2_zslice = clusterer.predict(feature_extractor.features)
        l2.append(
            l2_zslice.reshape(
                (
                    1,
                    slab_preshape[1],
                    slab_preshape[2],
                )
            )
        )

    sp_features = np.concatenate(sp_features, axis=0).reshape(
        (-1, feature_extraction_params["num_output_features"])
    )
    ic(sp_features.shape)

    l2 = np.concatenate(l2, axis=0)
    ic(l1.shape, l2.shape)
    np.save(
        "/data2/shreyas/mining_tomograms/working/tests/second_pass_features.npy",
        sp_features,
    )

    ic(np.allclose(old_features, sp_features))

    ic(np.allclose(l1, l2))
    np.save("/data2/shreyas/mining_tomograms/working/tests/l1.npy", l1)
    np.save("/data2/shreyas/mining_tomograms/working/tests/l2.npy", l2)


if __name__ == "__main__":
    main()
