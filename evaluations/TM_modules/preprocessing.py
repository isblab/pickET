import glob
import os
import yaml
import h5py
import mrcfile
import numpy as np
import re

from TM_modules.metadata import get_result_dirname


def numeric_key(path):
    filepath = os.path.abspath(path)
    return [
        int(x)
        if x.isdigit()
        else x for x in re.split(r"(\d+)", filepath)
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


def get_tomogram_files(
    tomogram_folder: str,
    dataset_type: str = "experimental"
):
    """ Discover tomograms in a given directory.

    Args:
        tomogram_folder (str): Directory to search for tomograms
        dataset_type (str, optional): Type of dataset
            "simulated" or "experimental" or "tutorial".
            Defaults to "experimental".

    Returns:
        list: List of discovered tomogram paths
    """

    if dataset_type in ["experimental", "tutorial"]:
        tomogram_files = sorted(
            glob.glob(os.path.join(tomogram_folder, "*.mrc")),
            key=numeric_key
        )
    elif dataset_type == "simulated": # tomotwin
        tomogram_files = glob.glob(
            os.path.join(tomogram_folder, "**", "*.mrc"),
            recursive=True
        )
        tomogram_files = sorted(
            [
                f for f in tomogram_files
                if os.path.basename(f) == "tiltseries_rec.mrc"
            ],
            key=numeric_key
        )
    else:
        raise ValueError(
            f"Invalid parameter dataset_type: {dataset_type}"
            "Valid options: 'experimental' or 'simulated' or 'tutorial'"
        )

    return tomogram_files


def run_preprocessing(
    tomogram_files: list,
    picket_in_h5: str,
    picket_out_mrc: str,
    tomogram_config: dict = {},
    dataset_type: str = "experimental",
):
    """ Preprocesing steps:
    - Convert h5 picket semantic segmenations to mrc format for each tomogram
    - Compute per-tomogram occupancy and save a summary file

    Args:
        tomogram_files (list): List of full paths to a set of tomograms
        picket_in_h5 (str): Picket semantic segmentation directory with *.h5 files
        picket_out_mrc (str): Output directory to store converted mrc segmentations
        tomogram_config (dict, optional): Parameters specific to each
            tomogram if any. Defaults to {}.
    """

    os.makedirs(picket_out_mrc, exist_ok=True)

    segmentation_files = sorted(
        glob.glob(os.path.join(picket_in_h5, "*.h5")),
        key=numeric_key
    )
    if len(segmentation_files) != len(tomogram_files):
        raise ValueError("Mismatch between tomograms and segmentations.")

    occupancy_summary = {}

    seg_iter = iter(segmentation_files)

    for tomo_file in tomogram_files:

        # tomo_name = os.path.splitext(os.path.basename(tomo_file))[0]
        tomo_name = get_result_dirname(
            dataset_type=dataset_type,
            tomogram_path=tomo_file,
        )
        tomo_cfg = tomogram_config.get(tomo_name, {})
        output_path = os.path.join(picket_out_mrc, f"{tomo_name}_mask.mrc")

        if "picket_segmentation" in tomo_cfg:
            h5_file = os.path.join(picket_in_h5, tomo_cfg["picket_segmentation"])
            if not os.path.exists(h5_file):
                raise FileNotFoundError(
                    f"Segmentation file not found for {tomo_name}: {h5_file}"
                )
        else:
            h5_file = next(seg_iter)
        print(f"{tomo_name} <-- {os.path.basename(h5_file)}")

        seg = load_h5_mask(h5_file)
        mask = convert_to_binary_mask(seg)
        inverted = False
        if tomo_cfg.get("invert_mask", False):
            mask = invert_mask(mask)
            inverted = True

        occupancy = compute_occupancy(mask)
        occupancy_summary[tomo_name] = {
            "occupancy_percent": round(occupancy, 2),
            "inverted": inverted,
            "segmentation_file": os.path.basename(h5_file),
        }
        print(
            f"{tomo_name}: " f"{occupancy:.2f}% " f"{'(inverted)' if inverted else ''}"
        )

        if os.path.exists(output_path):
            print(f"Skipping existing mask: {tomo_name}")
            continue

        save_mrc_mask(mask, output_path)

    summary_path = os.path.join(picket_out_mrc, "occupancy_summary.yaml")
    with open(summary_path, "w") as f:
        yaml.dump(occupancy_summary, f, sort_keys=False)

    print("\nSaved occupancy summary:")
    print(summary_path)
