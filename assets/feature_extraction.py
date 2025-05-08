import cupy as cp
import numpy as np


class FeatureExtractor:
    def __init__(self, window_size: int, feature_extraction_params: dict):
        self.window_size = window_size
        self.feature_extraction_params = feature_extraction_params
        self.mode = feature_extraction_params["mode"]
        self.features = np.array(None)
        self.best_feature_idxs = np.array([])

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
        return filter_bank

    def compute_ffts(self, windows: np.ndarray) -> np.ndarray:
        num_parallel_windows = (
            len(windows) // self.feature_extraction_params["n_fft_subsets"]
        )
        ffts = np.nan * np.ones(windows.shape)

        for subset_start_idx in range(0, len(windows), num_parallel_windows):
            subset_end_idx = min(
                (subset_start_idx + num_parallel_windows), len(windows)
            )
            subset = windows[subset_start_idx:subset_end_idx]
            subset_fft = cp.fft.fftn(cp.asarray(subset), axes=(-3, -2, -1))
            subset_fft = cp.fft.fftshift(subset_fft, axes=(-3, -2, -1))
            subset_fft = cp.abs(subset_fft)
            ffts[subset_start_idx:subset_end_idx] = cp.asnumpy(subset_fft)

        if np.any(np.isnan(ffts)):
            raise ValueError("Some FFTs were not computed")

        ffts = ffts.reshape((-1, self.window_size**3))
        return ffts

    def compute_gabor_features(self, windows: np.ndarray) -> np.ndarray:
        if len(windows.shape) > 2:
            raise ValueError("Windows must be 2D for Gabor feature extraction")
        if len(self.gabor_filter_bank.shape) > 2:
            raise ValueError(
                "Gabor filter bank must be 2D for Gabor feature extraction"
            )

        num_output_features = self.feature_extraction_params["num_output_features"]
        num_parallel_filters = self.feature_extraction_params["num_parallel_filters"]
        window_subset_size = (
            len(windows) // self.feature_extraction_params["num_windows_subsets"]
        )

        stds = np.nan * np.ones(len(self.gabor_filter_bank))

        if len(self.best_feature_idxs) == 0:
            for f_start_idx in range(
                0, len(self.gabor_filter_bank), num_parallel_filters
            ):
                f_end_idx = min(
                    len(self.gabor_filter_bank), f_start_idx + num_parallel_filters
                )
                f_subset = self.gabor_filter_bank[f_start_idx:f_end_idx]

                subset_features = cp.nan * cp.ones((len(windows), len(f_subset)))
                for w_start_idx in range(0, len(windows), window_subset_size):
                    w_end_idx = min(len(windows), w_start_idx + window_subset_size)
                    w_subset = windows[w_start_idx:w_end_idx]

                    f_subset = cp.asarray(f_subset, dtype=cp.float32)
                    w_subset = cp.asarray(w_subset, dtype=cp.float32)
                    ft = cp.matmul(w_subset, f_subset.T)
                    subset_features[w_start_idx:w_end_idx] = ft
                subset_stds = cp.asnumpy(cp.std(subset_features, axis=0))
                stds[f_start_idx:f_end_idx] = subset_stds

            self.best_feature_idxs = np.argsort(stds)[-num_output_features:]

        features = np.nan * np.ones((len(windows), num_output_features))
        for w_start_idx in range(0, len(windows), window_subset_size):
            w_end_idx = min(len(windows), w_start_idx + window_subset_size)
            w_subset = windows[w_start_idx:w_end_idx]

            filters = cp.asarray(
                self.gabor_filter_bank[self.best_feature_idxs], dtype=cp.float32
            )
            w_subset = cp.asarray(w_subset, dtype=cp.float32)
            ft = cp.matmul(w_subset, filters.T)
            features[w_start_idx:w_end_idx] = cp.asnumpy(ft)

        if np.any(np.isnan(features)):
            raise ValueError("Some Gabor features were not computed")

        return features

    def extract_features(self, windows: np.ndarray) -> None:
        if self.mode == "intensities":
            self.features = windows.reshape((-1, self.window_size**3))
        elif self.mode == "ffts":
            self.features = self.compute_ffts(windows)

        elif self.mode == "gabor":
            windows = windows.reshape((-1, self.window_size**3))
            self.features = self.compute_gabor_features(windows)

        else:
            raise ValueError(
                f"Feature extraction mode {self.mode} not supported.\
                             Choose from 'intensities', 'ffts', 'gabor'"
            )
