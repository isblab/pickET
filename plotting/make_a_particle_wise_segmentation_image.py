import os
import sys
import ndjson
import napari
import numpy as np
from rich.progress import track
from assets import utils, segmentation_io


def get_gt_coords(fpath: str) -> dict:
    with open(fpath, "r") as in_annot_f:
        annotations = ndjson.load(in_annot_f)
    out_dict = {}
    for ln in annotations:
        particle_name = ln["particle_name"]
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


def main():
    thickness = 5
    seg_path = sys.argv[1]
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

    label_ids = {
        "proton_transporting_atp_synthase_complex-1": 1,  # brown
        "mitochondrial_proton_transporting_atp_synthase_complex-1": 9,  # yellow
        "cytosolic_ribosome-1": 4,  # purple
        "ribulose_bisphosphate_carboxylase_complex-1": 5,
        "nucleosome-1": 10,  # red
        "fatty_acid_synthase_complex-1": 11,  # dark green
        "hydrogen_dependent_co2_reductase_filament-1": 12,
    }
    gt_annotations_path = os.path.join(
        "/", *tomo_path.split("/")[:-1], "annotations", "all_annotations.ndjson"
    )

    gt_annotations = get_gt_coords(gt_annotations_path)
    print(gt_annotations.keys())

    tomogram, _ = utils.load_tomogram(tomo_path)
    zslice_lb, zslice_ub = (
        tomogram.shape[0] // 2 - thickness,
        tomogram.shape[0] // 2 + thickness,
    )

    iseg = np.array(segmentation.instance_segmentation)

    masked_iseg = np.zeros(iseg.shape, dtype=np.int16)
    for k, v in track(gt_annotations.items()):
        for coord in v:
            z, y, x = coord
            if iseg[z, y, x] != 0 and zslice_lb < z < zslice_ub:
                idxs = np.where(iseg == iseg[z, y, x])
                masked_iseg[idxs] = label_ids[k]

    viewer = napari.Viewer()
    viewer.add_image(tomogram, name="Tomogram")
    viewer.add_labels(segmentation.semantic_segmentation, name="Semantic segmentation")
    viewer.add_labels(iseg, name="Instance segmentation")
    viewer.add_labels(masked_iseg, name="Masked instance segmentation")
    napari.run()
    print()


if __name__ == "__main__":
    main()
