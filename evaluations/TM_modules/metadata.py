import os
import mrcfile


def get_result_dirname(
    dataset_type: str,
    tomogram_path: str,
) -> str:

    if dataset_type == "simulated":
        basename = os.path.basename(os.path.dirname(tomogram_path))

    elif dataset_type in ["experimental", "tutorial"]:
        basename = os.path.splitext(os.path.basename(tomogram_path))[0]

    else:
        raise ValueError(
            f"Invalid parameter dataset_type: {dataset_type}"
            "Valid options: 'experimental' or 'simulated' or 'tutorial'"
        )

    return basename


def fix_tutorial_voxel_headers(tomogram_files: list):

    for fpath in tomogram_files:

        if not (fpath.endswith(".mrc")):
            continue

        with mrcfile.open(fpath, mode="r+", permissive=True) as mrc:

            mrc.voxel_size = 13.79


def discover_sidecar_files(mrc_path: str) -> list:

    base = os.path.splitext(mrc_path)[0]

    files = {"rawtlt": None}

    if os.path.exists(base + ".rawtlt"):

        files["rawtlt"] = base + ".rawtlt"

    return files


def get_metadata(
    tomogram_files: list,
    config: dict,
) -> list:

    if config["dataset"]["type"] == "tutorial":

        print("\nApplying tutorial " "voxel-size correction...")

        fix_tutorial_voxel_headers(tomogram_files=tomogram_files)

    dataset = []

    for _, mrc_file in enumerate(tomogram_files):

        result_name = get_result_dirname(
            dataset_type=config["dataset"]["type"],
            tomogram_path=mrc_file
        )

        tomo_cfg = config.get("tomograms", {}).get(result_name, {})

        unknown = set(tomo_cfg.keys()) - {
            "picket_segmentation",
            "invert_mask",
            "tilt_range",
        }

        if unknown:

            raise ValueError(f"{result_name}: unknown tomogram keys {unknown}")

        mask_path = os.path.join(
            config["preprocessing"]["picket_out_mrc"], f"{result_name}_mask.mrc"
        )

        # if not os.path.exists(mask_path):

        #     raise FileNotFoundError(f"Mask not found: {mask_path}")

        sidecars = discover_sidecar_files(mrc_path=mrc_file)

        # ----------------------------------
        # Tilt-angle handling
        # ----------------------------------

        if sidecars["rawtlt"] is not None:
            min_tilt = None
            max_tilt = None

        elif "tilt_range" in tomo_cfg:

            if len(tomo_cfg["tilt_range"]) != 2:
                raise ValueError(
                    f"{result_name}: tilt_range must contain [min, max]"
                )

            min_tilt = tomo_cfg["tilt_range"][0]
            max_tilt = tomo_cfg["tilt_range"][1]

        else:

            min_tilt = config["tilt_angles"]["min"]
            max_tilt = config["tilt_angles"]["max"]


        with mrcfile.open(mrc_file, permissive=True) as mrc:
            voxel_size = float(mrc.voxel_size.x)
            mrc_shape = mrc.data.shape

        tomogram = {
            "path": mrc_file,
            "shape": mrc_shape,
            "voxel_size": voxel_size,
            "rawtlt": sidecars["rawtlt"],
            "mask_path": mask_path,
            "min_tilt": min_tilt,
            "max_tilt": max_tilt,
        }

        dataset.append(tomogram)

    return dataset
