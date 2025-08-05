import os
import sys
import napari
import ndjson
import mrcfile
import numpy as np
from rich.progress import track
from picket.core import utils, segmentation_io


def get_gt_coords(fpath: str) -> dict:
    with open(fpath, "r") as in_annot_f:
        annotations = ndjson.load(in_annot_f)
    out_dict = {}
    for ln in annotations:
        particle_name = ln["particle_id"]

        if particle_name not in out_dict:
            out_dict[particle_name] = [
                np.array(
                    [
                        int(round(ln["location"]["z"], 0)),
                        int(round(ln["location"]["y"], 0)),
                        int(round(ln["location"]["x"], 0)),
                    ]
                )
            ]
        else:
            out_dict[particle_name].append(
                np.array(
                    [
                        int(round(ln["location"]["z"], 0)),
                        int(round(ln["location"]["y"], 0)),
                        int(round(ln["location"]["x"], 0)),
                    ]
                )
            )
    return out_dict


def mask_peripheral_voxels(segmentation: np.ndarray, window_size: int):
    peripheral_mask = np.zeros(segmentation.shape, dtype=np.int16)
    peripheral_mask[
        window_size // 2 : -window_size // 2 + 1,
        window_size // 2 : -window_size // 2 + 1,
        window_size // 2 : -window_size // 2 + 1,
    ] = 1
    return segmentation * peripheral_mask


chimerax_colors = {
    "-3": "#CDAB8F",
    "2": "#99C1F1",
    "3": "#DC8ADD",
    "4": "#F66151",
    "5": "#8FF0A4",
    "6": "#FF7800",
    "7": "#4C5EEC",
    "8": "#45F5EE",
    "9": "#C44850",
}


def main():
    seg_path = sys.argv[1]
    zl, zu = int(sys.argv[2]), int(sys.argv[3])
    out_dir = sys.argv[4]

    threshold = 0.5
    window_size = 5
    change_data_dir = True

    segmentation = segmentation_io.Segmentations()
    segmentation.load_segmentations(seg_path)

    if not change_data_dir:
        tomo_path = str(segmentation.metadata["tomogram_path"])
    else:
        tomo_path = str(segmentation.metadata["tomogram_path"]).replace(
            "data/", "data2/"
        )
    print(tomo_path)

    gt_annotations_path = os.path.join(
        "/", *tomo_path.split("/")[:-1], "coords", "all_annotations.ndjson"
    )
    gt_annotations = get_gt_coords(gt_annotations_path)

    tomogram, _ = utils.load_tomogram(tomo_path)

    semseg = np.array(segmentation.semantic_segmentation)
    iseg = np.array(segmentation.instance_segmentation)
    masked_iseg = np.zeros(iseg.shape, dtype=np.int16)

    for idx, (k, v) in enumerate(track(gt_annotations.items())):
        v = np.array(v)
        gt_z, gt_y, gt_x = v[:, 0], v[:, 1], v[:, 2]

        for z, y, x in zip(gt_z, gt_y, gt_x):
            gt_box = iseg[
                int(z - window_size) : int(z + window_size) + 1,
                int(y - window_size) : int(y + window_size) + 1,
                int(x - window_size) : int(x + window_size) + 1,
            ]
            proportion = np.count_nonzero(gt_box) / (window_size**3)

            if len(np.unique(gt_box)) == 2 and proportion > threshold:
                for label_val in np.unique(gt_box):
                    if label_val == 0:
                        continue
                    masked_iseg[np.where(iseg == label_val)] = idx

    c1 = semseg != 0
    c2 = masked_iseg == 0
    masked_iseg[np.where(c1 & c2)] = -3
    masked_iseg = mask_peripheral_voxels(masked_iseg, window_size)

    viewer = napari.Viewer()
    viewer.add_image(tomogram[zl:zu], name="Tomogram")
    viewer.add_labels(semseg[zl:zu], name="Semantic segmentation")
    viewer.add_labels(iseg[zl:zu], name="Instance segmentation")
    viewer.add_labels(masked_iseg[zl:zu], name="Masked instance segmentation")
    napari.run()

    for i in np.unique(masked_iseg):
        if i != 0:
            mrcmap = np.where(masked_iseg == i, 1, 0).astype(np.float32)[zl:zu]
            mrcfile.new(
                os.path.join(out_dir, f"masked_seg_{i}.mrc"),
                data=mrcmap,
                overwrite=True,
            )


if __name__ == "__main__":
    main()
