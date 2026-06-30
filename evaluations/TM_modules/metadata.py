import os
import mrcfile


def fix_tutorial_voxel_headers(dataset_path):
    
    for fname in os.listdir(dataset_path):
        
        if not (fname.endswith(".mrc")):
            continue

        mrc_path = os.path.join(dataset_path, fname)

        with mrcfile.open(mrc_path, mode="r+", permissive=True) as mrc:

            mrc.voxel_size = 13.79


def get_voxel_size(mrc_path):

    with mrcfile.open(mrc_path, permissive=True) as mrc:

        voxel_size = float(mrc.voxel_size.x)

    return voxel_size


def get_tomogram_shape(mrc_path):

    with mrcfile.open(mrc_path, permissive=True) as mrc:

        shape = mrc.data.shape

    return shape


def find_mrc_files(dataset_path):

    mrc_files = []

    for fname in os.listdir(dataset_path):

        if fname.endswith(".mrc"):

            mrc_files.append(os.path.join(dataset_path, fname))

    return sorted(mrc_files)


def discover_sidecar_files(mrc_path):

    base = os.path.splitext(mrc_path)[0]

    files = {"rawtlt": None}

    if os.path.exists(base + ".rawtlt"):

        files["rawtlt"] = base + ".rawtlt"

    return files


def get_metadata(dataset_path, config):

    if config["dataset"]["type"] == "tutorial":

        print(
            "\nApplying tutorial "
            "voxel-size correction..."
        )

        fix_tutorial_voxel_headers(dataset_path)

    dataset = []

    mrc_files = find_mrc_files(dataset_path)

    # ----------------------------------
    # Tomogram discovery
    # ----------------------------------

    for idx, mrc_file in enumerate(mrc_files):

        basename = os.path.splitext(os.path.basename(mrc_file))[0]

        tomo_cfg = config.get("tomograms", {}).get(basename, {})

        unknown = set(tomo_cfg.keys()) - {"segmentation", "invert_mask", "tilt_range"}

        if unknown:

            raise ValueError(f"{basename}: unknown tomogram keys {unknown}")

        mask_path = os.path.join(

            config["preprocessing"]["output_mask_folder"], f"{basename}_mask.mrc")

        if not os.path.exists(mask_path):

            raise FileNotFoundError(f"Mask not found: {mask_path}")

        sidecars = discover_sidecar_files(mrc_file)

        # ----------------------------------
        # Tilt-angle handling
        # ----------------------------------

        if sidecars["rawtlt"] is not None:
            min_tilt = None
            max_tilt = None

        elif "tilt_range" in tomo_cfg:

            if len(tomo_cfg["tilt_range"]) != 2:
                raise ValueError(f"{basename}: tilt_range must contain [min, max]")

            min_tilt = tomo_cfg["tilt_range"][0]
            max_tilt = tomo_cfg["tilt_range"][1]

        else:

            min_tilt = config["tilt_angles"]["min"]
            max_tilt = config["tilt_angles"]["max"]

        tomogram = {

            "path": mrc_file,

            "shape": get_tomogram_shape(mrc_file),

            "voxel_size": get_voxel_size(mrc_file),

            "rawtlt": sidecars["rawtlt"],

            "mask_path": mask_path,

            "min_tilt": min_tilt,

            "max_tilt": max_tilt

        }

        dataset.append(tomogram)

    return dataset
