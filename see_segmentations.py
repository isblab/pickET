import os
import sys
import napari
import numpy as np
from assets import utils, segmentation_io


def main():
    seg_path = sys.argv[1]
    seg_type = sys.argv[2]
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

    tomo_path: str = str(segmentation_metadata["tomogram_path"])

    #! Deprecated: Remove this section later on
    tomo_path2 = tomo_path.split("/")[2:]
    tomo_path = os.path.join("/data2", *tomo_path2)

    tomogram, _ = utils.load_tomogram(tomo_path)
    print("Loaded segmentation and tomogram successfully...")

    viewer = napari.Viewer()
    viewer.add_image(tomogram, name="Tomogram")
    viewer.add_labels(segmentation, name="Segmentation")
    napari.run()


if __name__ == "__main__":
    main()
