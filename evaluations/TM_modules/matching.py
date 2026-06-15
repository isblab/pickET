import os
import subprocess


def build_tm_command(
    tomo,
    template_path,
    mask_path,
    particle_diameter,
    config,
    results_dir
):

    cmd = [

        "pytom_match_template.py",

        "-t",
        template_path,

        "-m",
        mask_path,

        "-v",
        tomo["path"],

        "-d",
        results_dir,

        "--particle-diameter",
        str(
            particle_diameter
        ),

        "--low-pass",
        "40"
    ]

    # ---------------------------------
    # Rotational symmetry
    # ---------------------------------

    rot_sym = (

        config["template_matching"]
              ["rotational_symmetry"]

    )

    if rot_sym > 1:

        cmd.extend([

            "--z-axis-rotational-symmetry",

            str(rot_sym)

        ])

    # ---------------------------------
    # Tilt angles
    # ---------------------------------

    if tomo["rawtlt"] is not None:

        cmd.extend([

            "-a",

            tomo["rawtlt"]

        ])

    else:

        cmd.extend([

            "-a",

            str(
                config["tilt_angles"]["min"]
            ),

            str(
                config["tilt_angles"]["max"]
            )

        ])

    # ---------------------------------
    # Volume split
    # ---------------------------------

    cmd.extend([

        "-s",

        str(
            config["compute"]
                  ["volume_split"]["x"]
        ),

        str(
            config["compute"]
                  ["volume_split"]["y"]
        ),

        str(
            config["compute"]
                  ["volume_split"]["z"]
        )

    ])

    # ---------------------------------
    # Search region
    # ---------------------------------

    if config["search_region"]["enabled"]:

        cmd.extend([

            "--search-x",

            str(
                config["search_region"]
                      ["x"][0]
            ),

            str(
                config["search_region"]
                      ["x"][1]
            )

        ])

        cmd.extend([

            "--search-y",

            str(
                config["search_region"]
                      ["y"][0]
            ),

            str(
                config["search_region"]
                      ["y"][1]
            )

        ])

        cmd.extend([

            "--search-z",

            str(
                config["search_region"]
                      ["z"][0]
            ),

            str(
                config["search_region"]
                      ["z"][1]
            )

        ])

    # ---------------------------------
    # Tomogram mask
    # ---------------------------------

    if (
        config["tomogram_mask"]["enabled"]
        and
        tomo["mask_path"] is not None
       ):

        cmd.extend([

            "--tomogram-mask",

            tomo["mask_path"]

        ])

    # ---------------------------------
    # CTF
    # ---------------------------------

    if config["ctf"]["enabled"]:

        cmd.extend([

            "--defocus",

            str(
                config["ctf"]["defocus"]
            )

        ])

        cmd.extend([

            "--amplitude-contrast",

            str(
                config["ctf"]
                      ["amplitude_contrast"]
            )

        ])

        cmd.extend([

            "--spherical-aberration",

            str(
                config["ctf"]
                      ["spherical_aberration"]
            )

        ])

        cmd.extend([

            "--voltage",

            str(
                config["ctf"]
                      ["voltage"]
            )

        ])

        if config["ctf"]["tomogram_ctf_model"] is not None:
            cmd.extend([
                "--tomogram-ctf-model",
                config["ctf"]["tomogram_ctf_model"]

            ])

        if config["ctf"]["per_tilt_weighting"]:

            cmd.extend([

                "--per-tilt-weighting"

            ])

    # ---------------------------------
    # Random phase correction
    # ---------------------------------

    if config["template_matching"][
        "random_phase_correction"
    ]:

        cmd.append("-r")

    # ---------------------------------
    # GPU IDs
    # ---------------------------------

    cmd.append("-g")

    for gpu in config["compute"]["gpu_ids"]:

        cmd.append(str(gpu))

    return cmd

def run_tm_command(cmd):

    print("\nRunning TM:\n")

    print(" ".join(cmd))

    subprocess.run(
        cmd,
        check=True
    )
