import subprocess
import os


def count_particles(star_file):

    if not os.path.exists(star_file):
        return 0

    count = 0

    with open(star_file) as f:

        for line in f:

            line = line.strip()

            if (
                not line
                or line.startswith("#")
                or line.startswith("data_")
                or line.startswith("loop_")
                or line.startswith("_")
            ):
                continue

            count += 1

    return count
    
def build_roc_command(
    job_file,
    config,
    extraction_particle_diameter,
    number_of_particles,
    ignore_tomogram_mask = False
):

    cmd = [

        "pytom_estimate_roc.py",

        "-j",
        job_file,

        "-n",
        str(number_of_particles),

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
