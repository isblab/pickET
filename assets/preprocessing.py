import mrcfile
import numpy as np
import scipy.ndimage as nd
from typing import Optional


def load_tomogram(tomogram_path: str) -> np.ndarray:
    return mrcfile.read(tomogram_path)


def get_z_section(
    tomogram: np.ndarray, z_lb: Optional[int] = None, z_ub: Optional[int] = None
) -> np.ndarray:
    return tomogram[z_lb:z_ub]


def guassian_blur_tomogram(tomogram: np.ndarray, sigma: float = 2) -> np.ndarray:
    return nd.gaussian_filter(tomogram, sigma=sigma)


def minmax_normalize_array(in_array: np.ndarray) -> np.ndarray:
    return (in_array - np.min(in_array)) / (np.max(in_array) - np.min(in_array))
