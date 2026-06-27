import glob
import os
import yaml
import h5py
import mrcfile
import numpy as np
import re

def numeric_key(path):
    filename = os.path.basename(path)
    return [
        int(x) if x.isdigit() else x
        for x in re.split(r'(\d+)', filename)
    ]

def load_h5_mask(h5_file):

    with h5py.File(h5_file, "r") as f:
        seg = f["segmentations"]["semantic_segmentation"][:]
    return seg

def convert_to_binary_mask(seg):

    return (seg == 1).astype(np.uint8)

def invert_mask(mask):

    return (1 - mask).astype(np.uint8)

def compute_occupancy(mask):

    return float(100 * np.mean(mask > 0))

def save_mrc_mask(mask, output_path):

    with mrcfile.new(output_path, overwrite=True) as m:
        m.set_data(mask)

def run_preprocessing(
    tomogram_folder,
    segmentation_folder,
    output_mask_folder,
    invert_masks=None
):

    if invert_masks is None:
        invert_masks = []
    os.makedirs(output_mask_folder, exist_ok=True)
    segmentation_files = sorted(
        glob.glob(os.path.join(segmentation_folder, "*.h5")), key=numeric_key)
    tomogram_files = sorted(
        glob.glob(os.path.join(tomogram_folder, "*.mrc")), key=numeric_key)
    if len(segmentation_files) != len(tomogram_files):
        raise ValueError("Mismatch between tomograms and segmentations.")

    occupancy_summary = {}

    for tomo_file, h5_file in zip(tomogram_files, segmentation_files):
        print(f"{os.path.basename(tomo_file)} <-- {os.path.basename(h5_file)}")
        tomo_name = os.path.splitext(
            os.path.basename(tomo_file))[0]
        seg = load_h5_mask(h5_file)
        mask = convert_to_binary_mask(seg)
        inverted = False
        if tomo_name in invert_masks:
            mask = invert_mask(mask)
            inverted = True
        occupancy = compute_occupancy(mask)
        output_path = os.path.join(
            output_mask_folder, f"{tomo_name}_mask.mrc")
        occupancy_summary[
            tomo_name
        ] = {
            "occupancy_percent": round(occupancy, 2),
            "inverted": inverted,
            "segmentation_file": os.path.basename(h5_file)
        }
        print(
            f"{tomo_name}: "
            f"{occupancy:.2f}% "
            f"{'(inverted)' if inverted else ''}"
        )

        if os.path.exists(output_path):
            print(f"Skipping existing mask: {tomo_name}")
            continue
        save_mrc_mask(mask, output_path)

    summary_path = os.path.join(output_mask_folder, "occupancy_summary.yaml")
    with open(summary_path, "w") as f:
        yaml.dump(occupancy_summary, f, sort_keys=False)
    print("\nSaved occupancy summary:")
    print(summary_path)
