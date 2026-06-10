import os
import mrcfile

def fix_tutorial_voxel_headers(
    dataset_path
):
    for fname in os.listdir(
        dataset_path
    ):
        if not (
            fname.endswith(".mrc")
        ):
            continue

        mrc_path = os.path.join(
            dataset_path,
            fname
        )

        with mrcfile.open(
            mrc_path,
            mode="r+",
            permissive=True
        ) as mrc:

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
            mrc_files.append(
                os.path.join(dataset_path, fname)
            )

    return sorted(mrc_files)

def discover_sidecar_files(mrc_path):

    base = os.path.splitext(mrc_path)[0]
    files = {
        "rawtlt": None,
        "defocus": None,
        "dose": None
    }

    if os.path.exists(base + ".rawtlt"):
        files["rawtlt"] = base + ".rawtlt"

    if os.path.exists(base + ".defocus"):
        files["defocus"] = base + ".defocus"

    if os.path.exists(base + "_dose.txt"):
        files["dose"] = base + "_dose.txt"

    return files

def get_metadata(dataset_path, config):

    if config["dataset"][
        "type"
    ] == "tutorial":

        print("\nApplying tutorial voxel-size correction...")

        fix_tutorial_voxel_headers(dataset_path)

    dataset = []
    mrc_files = find_mrc_files(dataset_path)

    for mrc_file in mrc_files:

        basename = os.path.splitext(
            os.path.basename(mrc_file)
        )[0]

        mask_path = None

        if config["tomogram_mask"]["enabled"]:
            mask_path = os.path.join(
                config["tomogram_mask"]["directory"],
                f"{basename}_mask.mrc"
            )

        sidecars = discover_sidecar_files(mrc_file)

        tomogram = {
            "path": mrc_file,
            "shape": get_tomogram_shape(mrc_file),
            "voxel_size": get_voxel_size(mrc_file),
            "rawtlt": sidecars["rawtlt"],
            "defocus": sidecars["defocus"],
            "dose": sidecars["dose"],
            "mask_path": mask_path
        }

        dataset.append(tomogram)

    return dataset
