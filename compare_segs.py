import sys
import numpy as np
from picket.core import segmentation_io


def main():
    seg1_path = sys.argv[1]
    seg2_path = sys.argv[2]

    seg1 = segmentation_io.Segmentations()
    seg1.load_segmentations(seg1_path)
    seg2 = segmentation_io.Segmentations()
    seg2.load_segmentations(seg2_path)

    true_val_count = np.sum(
        (seg1.semantic_segmentation == seg2.semantic_segmentation).astype(np.int32)
    )
    total = (
        np.array(seg1.semantic_segmentation).shape[0]
        * np.array(seg1.semantic_segmentation).shape[1]
        * np.array(seg1.semantic_segmentation).shape[2]
    )
    print(true_val_count, total, true_val_count / total)


if __name__ == "__main__":
    main()
