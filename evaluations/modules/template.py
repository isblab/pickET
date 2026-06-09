import subprocess
import mrcfile


def get_template_voxel_size(template_path):

    with mrcfile.open(
        template_path,
        permissive=True
    ) as mrc:

        voxel_size = float(
            mrc.voxel_size.x
        )

    return voxel_size


def generate_template(
    template_path,
    output_template,
    input_voxel_size,
    output_voxel_size,
    box_size,
    invert,
):

    cmd = [

        "pytom_create_template.py",

        "-i",
        template_path,

        "-o",
        output_template,

        "--input-voxel-size",
        str(input_voxel_size),

        "--output-voxel-size",
        str(output_voxel_size),

        "--box-size",
        str(box_size)

    ]

    if invert:

        cmd.append("--invert")

    cmd.append("--center")

    print("\nRunning:")

    print(" ".join(cmd))

    subprocess.run(
        cmd,
        check=True
    )


def generate_mask(
    box_size,
    radius,
    output_mask
):

    cmd = [

        "pytom_create_mask.py",

        "-b",
        str(box_size),

        "-r",
        str(radius),

        "--sigma",
        "1",

        "-o",
        output_mask
    ]

    print("\nRunning:")

    print(" ".join(cmd))

    subprocess.run(
        cmd,
        check=True
    )
