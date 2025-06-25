import os
import sys
import napari
import ndjson
import mrcfile
import numpy as np
from picket.core import utils, segmentation_io


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


def mask_peripheral_voxels(segmentation: np.ndarray, window_size: int):
    peripheral_mask = np.zeros(segmentation.shape, dtype=np.int16)
    peripheral_mask[
        window_size // 2 : -window_size // 2 + 1,
        window_size // 2 : -window_size // 2 + 1,
        window_size // 2 : -window_size // 2 + 1,
    ] = 1
    return segmentation * peripheral_mask


def main():
    seg_path = sys.argv[1]
    zl, zu = int(sys.argv[2]), int(sys.argv[3])
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

    label_ids = {
        "UNKNOWS": -3,  # #CDAB8F
        "proton_transporting_atp_synthase_complex-1": 1,  # #F5C211
        "mitochondrial_proton_transporting_atp_synthase_complex-1": 9,  # #F5C211
        "cytosolic_ribosome-1": 4,  # #99C1F1
        "ribulose_bisphosphate_carboxylase_complex-1": 5,  # #DC8ADD
        "nucleosome-1": 10,  # #F66151
        "fatty_acid_synthase_complex-1": 11,  # #8FF0A4
        "hydrogen_dependent_co2_reductase_filament-1": 12,  #
    }
    gt_annotations_path = os.path.join(
        "/", *tomo_path.split("/")[:-1], "annotations", "all_annotations.ndjson"
    )

    gt_annotations = get_gt_coords(gt_annotations_path)

    tomogram, _ = utils.load_tomogram(tomo_path)

    semseg = np.array(segmentation.semantic_segmentation)
    iseg = np.array(segmentation.instance_segmentation)
    masked_iseg = np.zeros(iseg.shape, dtype=np.int16)
    label_ids_used = []

    for k, v in gt_annotations.items():
        v = np.array(v)
        gt_z, gt_y, gt_x = v[:, 0], v[:, 1], v[:, 2]
        nonzero_idxs = np.where(iseg[gt_z, gt_y, gt_x] != 0)[0]
        gt_z, gt_y, gt_x = gt_z[nonzero_idxs], gt_y[nonzero_idxs], gt_x[nonzero_idxs]

        target_iseg_vals = iseg[gt_z, gt_y, gt_x]
        masked_iseg[np.isin(iseg, target_iseg_vals)] = label_ids[k]
        label_ids_used.append(label_ids[k])

    # semseg = np.where(iseg > 0, -3, 0)
    c1 = masked_iseg == 0
    c2 = semseg != 0
    masked_iseg = np.where(c1 & c2, -3, masked_iseg)

    masked_iseg = mask_peripheral_voxels(masked_iseg, window_size)
    iseg = mask_peripheral_voxels(iseg, window_size)

    viewer = napari.Viewer()
    viewer.add_image(tomogram[zl:zu], name="Tomogram")
    viewer.add_labels(semseg[zl:zu], name="Semantic segmentation")
    viewer.add_labels(iseg[zl:zu], name="Instance segmentation")
    viewer.add_labels(masked_iseg[zl:zu], name="Masked instance segmentation")
    napari.run()
    print()

    for i in np.unique(masked_iseg):
        out_dir = "./"
        if i != 0:
            mrcmap = np.where(masked_iseg == i, 1, 0).astype(np.float32)[zl:zu]
            mrcfile.new(
                os.path.join(out_dir, f"masked_seg_{i}.mrc"),
                data=mrcmap,
                overwrite=True,
            )


if __name__ == "__main__":
    main()
