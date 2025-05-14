import sys
import napari
import numpy as np
from assets import utils, segmentation_io


def main():
    seg_path = sys.argv[1]

    segmentation_handler = segmentation_io.Segmentations()
    segmentation_handler.load_segmentations(seg_path)
    segmentation = np.array(segmentation_handler.semantic_segmentation)
    segmentation_metadata = segmentation_handler.metadata
    tomo_path: str = str(segmentation_metadata["tomogram_path"])
    tomogram, _ = utils.load_tomogram(tomo_path)
    print("Loaded segmentation and tomogram successfully...")

    viewer = napari.Viewer()
    viewer.add_image(tomogram, name="Tomogram")
    viewer.add_labels(segmentation, name="Segmentation")
    napari.run()


if __name__ == "__main__":
    main()
