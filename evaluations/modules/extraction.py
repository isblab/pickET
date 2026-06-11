import subprocess


def build_extraction_command(
    job_file,
    config,
    extraction_particle_diameter,
    ignore_tomogram_mask = False
):

    cmd = [

        "pytom_extract_candidates.py",

        "-j",
        job_file,

        "-n",
        str(
            config["extraction"]
                  ["number_of_particles"]
        )

    ]

    cutoff = (
        config["extraction"]
              ["cutoff"]
    )

    cmd.extend([

            "--particle-diameter",

            str(extraction_particle_diameter)

    ])

    if cutoff is not None:

        cmd.extend([

            "-c",

            str(cutoff)

        ])


    if ignore_tomogram_mask:
        cmd.append(
            "--ignore_tomogram_mask"
        )

    return cmd


def run_extraction_command(cmd):

    print(
        "\nRunning Extraction:\n"
    )

    print(
        " ".join(cmd)
    )

    subprocess.run(
        cmd,
        check=True
    )
