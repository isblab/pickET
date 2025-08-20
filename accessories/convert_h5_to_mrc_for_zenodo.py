import os
import sys
import mrcfile
import numpy as np
from picket.core import segmentation_io


def main():
    seg_path = sys.argv[1]
    out_dir = sys.argv[2]

    segmentation_handler = segmentation_io.Segmentations()
    segmentation_handler.load_segmentations(seg_path)
    segmentation_metadata = segmentation_handler.metadata
    semantic_segmentation = np.array(segmentation_handler.semantic_segmentation)
    instance_segmentation = np.array(segmentation_handler.instance_segmentation)

    seg_idx = seg_path.split("/")[-1].split("_")[2]
    out_semseg_fname = os.path.join(
        out_dir,
        f"{segmentation_metadata['dataset_name']}_{seg_idx}_semantic_segmentation.mrc",
    )
    out_iseg_fname = os.path.join(
        out_dir,
        f"{segmentation_metadata['dataset_name']}_{seg_idx}_instance_segmentation.mrc",
    )

    print(segmentation_metadata["tomogram_path"])

    mrcfile.new(
        out_semseg_fname, semantic_segmentation.astype(np.float32), overwrite=True
    )
    mrcfile.new(
        out_iseg_fname, instance_segmentation.astype(np.float32), overwrite=True
    )


if __name__ == "__main__":
    main()
