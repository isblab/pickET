import subprocess


def build_roc_command(
    job_file,
    config,
    extraction_particle_diameter,
    ignore_tomogram_mask = False
):

    cmd = [

        "pytom_estimate_roc.py",

        "-j",
        job_file,

        "-n",
        str(
            config["roc"]
                  ["number_of_particles"]
        ),

        "--particle-diameter",
        str(extraction_particle_diameter),

        "--bins",
        str(
            config["roc"]
                  ["bins"]
        )

    ]

    if config["roc"]["crop_plot"]:

        cmd.append(
            "--crop-plot"
        )

    if ignore_tomogram_mask:

        cmd.append(
            "--ignore_tomogram_mask"
        )

    return cmd


def run_roc_command(
    cmd,
    log_file
):

    print(
        "\nRunning ROC:\n"
    )

    print(
        " ".join(cmd)
    )

    with open(
        log_file,
        "w"
    ) as f:

        subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True
        )
