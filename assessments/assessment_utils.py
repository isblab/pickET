import mrcfile
import numpy as np


def get_voxel_threshold(fname: str, angs_threshold: float) -> int:
    with mrcfile.open(fname, mode="r", permissive=True) as mrcf:
        voxel_sizes = np.array(
            [mrcf.voxel_size.z, mrcf.voxel_size.y, mrcf.voxel_size.x], dtype=np.float32
        )

    voxel_size = np.max(voxel_sizes)
    threshold = int(round(angs_threshold / voxel_size, 0))
    return threshold
