import os
import sys
import numpy as np
from assets import utils, segmentation_io


def fix_tomo_path(tomo_path: str, dataset_dir_drive: str) -> str:
    tomo_path_broken = tomo_path.split("/")
    tomo_path_broken[1] = dataset_dir_drive
    tomo_path = os.path.join("/", *tomo_path_broken)
    return tomo_path


def main():
    seg_path = sys.argv[1]
    seg_type = sys.argv[2]
    dataset_dir_drive = sys.argv[3]

    if seg_type not in ("semantic_segmentation", "instance_segmentation"):
        raise ValueError(
            "Segmentation type can only be either 'semantic_segmentation' or 'instance_segmentation'"
        )

    segmentation_handler = segmentation_io.Segmentations()
    segmentation_handler.load_segmentations(seg_path)
    segmentation_metadata = segmentation_handler.metadata

    if seg_type == "semantic_segmentation":
        segmentation = np.array(segmentation_handler.semantic_segmentation)
    else:
        segmentation = np.array(segmentation_handler.instance_segmentation)

    tomo_path = str(segmentation_metadata["tomogram_path"])

    if tomo_path.split("/")[1] != dataset_dir_drive:
        tomo_path = fix_tomo_path(tomo_path, dataset_dir_drive)

    tomogram, _ = utils.load_tomogram(tomo_path)
    print("Loaded segmentation and tomogram successfully...")
    utils.load_in_napari(tomogram, segmentation, segname=seg_type)


if __name__ == "__main__":
    main()
