import mrcfile
import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import gaussian_filter
from typing import Optional


def load_tomogram(tomogram_path: str) -> tuple[np.ndarray, np.ndarray]:
    with mrcfile.open(tomogram_path, "r", permissive=True) as mrcf:
        tomogram = np.array(mrcf.data)
        vxs = mrcf.voxel_size
        voxel_sizes = np.zeros(3, dtype=np.float32)
        for i, ax in enumerate(("z", "y", "x")):
            voxel_sizes[i] = vxs[ax]
    return tomogram, voxel_sizes


def get_z_section(
    tomogram: np.ndarray, z_lb: Optional[int] = None, z_ub: Optional[int] = None
) -> np.ndarray:
    return tomogram[z_lb:z_ub]


def guassian_blur_tomogram(tomogram: np.ndarray, sigma: float = 2) -> np.ndarray:
    return cp.asnumpy(gaussian_filter(cp.asarray(tomogram), sigma=sigma))


def minmax_normalize_array(in_array: np.ndarray) -> np.ndarray:
    return (in_array - np.min(in_array)) / (np.max(in_array) - np.min(in_array))
