import torch
import numpy as np
from tqdm import tqdm
from skimage.util import view_as_windows


class FeatureExtractor:
    def __init__(self, window_size: int, feature_extraction_params: dict):
        self.window_size = window_size
        self.feature_extraction_params = feature_extraction_params
        self.windows = np.array(None)
        self.preshape = np.array(None)
        self.mode = feature_extraction_params.get("mode")
        self.features = np.array(None)

        if self.mode == "gabor":
            self.gabor_filter_bank = self.generate_gabor_filter_bank()

    def generate_gabor_filter_bank(self) -> np.ndarray:
        def get_gabor_filter(freqs: list, psi: float = 0) -> np.ndarray:
            def get_gaussian():
                half_size = self.window_size // 2
                sigma = half_size / 3
                (z, y, x) = np.meshgrid(
                    np.arange(-half_size, half_size + 1),
                    np.arange(-half_size, half_size + 1),
                    np.arange(-half_size, half_size + 1),
                )
                gaussian = np.exp(
                    -0.5 * ((x**2 + y**2 + z**2) / (2 * sigma**2))
                )  # Using a global sigma
                return gaussian

            z, y, x = np.meshgrid(
                np.linspace(0, self.window_size, self.window_size),
                np.linspace(0, self.window_size, self.window_size),
                np.linspace(0, self.window_size, self.window_size),
            )
            wave = np.cos(
                2 * np.pi * (freqs[0] * z + freqs[1] * y + freqs[2] * x) + psi
            )
            gaussian = wave * get_gaussian()
            gabor_filter = wave * gaussian
            gabor_filter /= np.sum(gabor_filter)
            return gabor_filter

        num_sinusoids = self.feature_extraction_params["num_sinusoids"]
        max_freq = 0.5  # in Cycles/pixel as per the Nyquist-Shannon sampling theorem
        filter_bank = np.nan * np.ones((num_sinusoids**3, self.window_size**3))

        idx = 0
        for freq_z in np.linspace(0, max_freq, num_sinusoids):
            for freq_y in np.linspace(0, max_freq, num_sinusoids):
                for freq_x in np.linspace(0, max_freq, num_sinusoids):
                    gabor_filter = get_gabor_filter(freqs=[freq_z, freq_y, freq_x])
                    filter_bank[idx] = gabor_filter.reshape((self.window_size**3))
                    idx += 1
        if np.any(np.isnan(filter_bank)):
            raise ValueError("Some Gabor filters were not generated")
        print(f"Generated a Gabor filter bank with {len(filter_bank)} filters\n")
        return filter_bank

    def get_windows(self, tomo: np.ndarray) -> np.ndarray:
        if (not (isinstance(self.window_size, int))) or self.window_size % 2 == 0:
            raise ValueError(
                f"Please set window_size to an odd integer. \
                    It was set to {self.window_size} of type {type(self.window_size)}"
            )
        windows = view_as_windows(tomo, self.window_size)
        self.preshape = windows.shape[:3]
        self.windows = windows.reshape(
            -1, self.window_size, self.window_size, self.window_size
        )
        return windows

    def compute_ffts(self) -> np.ndarray:
        ffts = np.nan * np.ones(self.windows.shape)
        window_subsets = np.array_split(
            self.windows, self.feature_extraction_params["n_fft_subsets"]
        )

        writer_idx = 0
        for subset in tqdm(window_subsets, desc="Computing FFTs"):
            subset_fft = np.fft.fftn(subset, axes=(1, 2, 3))
            subset_fft = np.fft.fftshift(subset_fft, axes=(1, 2, 3))
            subset_fft = np.abs(subset_fft)
            ffts[writer_idx : writer_idx + len(subset)] = subset_fft
            writer_idx += len(subset)

        if np.any(np.isnan(ffts)):
            raise ValueError("Some FFTs were not computed")

        ffts = np.where(ffts < np.mean(ffts), 0, ffts)
        ffts = ffts.reshape((-1, self.window_size**3))
        return ffts

    def get_highest_std_features(
        self, features: np.ndarray, n_output_features: int, use_subset_size: int
    ) -> np.ndarray:
        print(
            "\tExtracting the features with the highest standard deviation in the tomogram"
        )
        random_idx = np.random.choice(np.arange(len(features) - 1), use_subset_size)
        stds = np.std(features[random_idx], axis=0)

        top_std = np.argsort(stds)[-n_output_features:]
        dim_reduced_features = features[:, top_std]
        dim_reduced_features = np.squeeze(dim_reduced_features)
        return dim_reduced_features

    def compute_gabor_features(self) -> np.ndarray:
        if len(self.windows.shape) > 2:
            raise ValueError("Windows must be 2D for Gabor feature extraction")
        if len(self.gabor_filter_bank.shape) > 2:
            raise ValueError(
                "Gabor filter bank must be 2D for Gabor feature extraction"
            )

        device = self.feature_extraction_params.get("device", "cpu")
        num_output_features = self.feature_extraction_params["num_output_features"]
        num_parallel_filters = self.feature_extraction_params["num_parallel_filters"]
        replacement_mode = False

        filter_subsets = np.array_split(
            self.gabor_filter_bank,
            np.arange(
                num_parallel_filters, len(self.gabor_filter_bank), num_parallel_filters
            ),
        )
        window_subsets = np.array_split(
            self.windows, self.feature_extraction_params["num_windows_subsets"]
        )

        features = np.nan * np.ones(
            (
                len(self.windows),
                num_output_features + num_parallel_filters,
            )
        )
        stds = np.nan * np.ones((num_output_features + num_parallel_filters))

        for f_idx, f_subset in enumerate(tqdm(filter_subsets)):
            counter = min(
                f_idx * len(filter_subsets[0]),
                num_output_features,
            )

            subset_features = torch.nan * torch.ones((len(self.windows), len(f_subset)))
            for w_idx, w_subset in enumerate(window_subsets):
                f_subset = torch.tensor(f_subset, dtype=torch.float32, device=device)
                w_subset = torch.tensor(w_subset, dtype=torch.float32, device=device)
                ft = torch.matmul(w_subset, f_subset.T)
                subset_features[
                    w_idx * len(window_subsets[0]) : w_idx * len(window_subsets[0])
                    + len(w_subset)
                ] = ft
            subset_stds = torch.std(subset_features, dim=0)

            if not replacement_mode:
                features[:, counter : counter + len(f_subset)] = (
                    subset_features.cpu().numpy()
                )
                stds[counter : counter + len(f_subset)] = subset_stds.cpu().numpy()
            else:
                features = np.concatenate(
                    (features, subset_features.cpu().numpy()), axis=1
                )
                stds = np.concatenate((stds, subset_stds.cpu().numpy()))

            if counter + len(f_subset) > num_output_features:
                replacement_mode = True
                best_idxs = np.argsort(stds)[::-1][:num_output_features]
                features = features[:, best_idxs]
                stds = stds[best_idxs]

            del subset_features, f_subset, w_subset, subset_stds, ft

        if np.any(np.isnan(features)):
            raise ValueError("Some Gabor features were not computed correctly")

        return features

    def extract_features(self) -> None:
        if self.mode == "intensities":
            self.features = self.windows.reshape((-1, self.window_size**3))
        elif self.mode == "ffts":
            self.features = self.get_highest_std_features(
                self.compute_ffts(),
                self.feature_extraction_params["num_output_features"],
                self.feature_extraction_params["use_subset_size"],
            )
        elif self.mode == "gabor":
            self.windows = self.windows.reshape((-1, self.window_size**3))
            self.features = self.compute_gabor_features()

        else:
            raise ValueError(
                f"Feature extraction mode {self.mode} not supported.\
                             Choose from 'intensities', 'ffts', 'gabor'"
            )
