import sys
import napari
from assets import utils, preprocessing


def main():
    seg_path = sys.argv[1]

    segmentation, segmentation_metadata = utils.load_h5file(seg_path)  # type:ignore
    tomo_path: str = segmentation_metadata["tomogram_path"]
    tomogram, _ = utils.load_tomogram(tomo_path)

    viewer = napari.Viewer()
    viewer.add_image(tomogram, name="Tomogram")
    viewer.add_labels(segmentation, name="Segmentation")
    napari.run()


if __name__ == "__main__":
    main()
