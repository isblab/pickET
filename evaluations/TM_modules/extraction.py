import subprocess
import numpy as np
from Bio.PDB import PDBParser

def get_masses_and_coords(pdb_file):

    parser = PDBParser(QUIET=True)

    structure = parser.get_structure("structure", pdb_file)

    masses = []
    coords = []

    for atom in structure.get_atoms():

        masses.append(atom.mass)

        coords.append(atom.coord)

    return (np.array(masses), np.array(coords))


def get_radius_of_gyration(pdb_file):

    masses, coords = get_masses_and_coords(pdb_file)

    total_mass = np.sum(masses)

    weighted_coords = coords.T * masses

    center_of_mass = np.sum(weighted_coords, axis=1) / total_mass

    squared_distances = np.sum((coords - center_of_mass) ** 2, axis=1)

    rg = np.sqrt(np.sum(squared_distances * masses) / total_mass)

    return float(rg)


def get_extraction_diameter(pdb_file):

    rg = get_radius_of_gyration(pdb_file)

    return int(round(2 * rg))


def build_extraction_command(
    job_file,
    config,
    extraction_particle_diameter,
    tomogram_mask=None
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

    if tomogram_mask is not None:
        cmd.extend([
            "--tomogram-mask",
            tomogram_mask
        ])

    cmd.append("--relion5-compat")

    return cmd


def run_extraction_command(cmd):

    print("\nRunning Extraction:\n")

    print(" ".join(cmd))

    subprocess.run(cmd, check=True)
