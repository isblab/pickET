import subprocess
import mrcfile
import numpy as np
from scipy.spatial.distance import cdist
from Bio.PDB import PDBParser


def get_template_voxel_size(template_path):

    with mrcfile.open(
        template_path,
        permissive=True
    ) as mrc:

        voxel_size = float(
            mrc.voxel_size.x
        )

    return voxel_size


def get_atom_coordinates(
    pdb_file
):

    parser = PDBParser(QUIET=True)

    structure = parser.get_structure(
        "structure",
        pdb_file
    )

    coordinates = []

    for model in structure:

        for chain in model:

            for residue in chain:

                for atom in residue:

                    coordinates.append(
                        atom.get_coord()
                    )

    return np.array(
        coordinates
    )

def get_masses_and_coords(
    pdb_file
):

    parser = PDBParser(
        QUIET=True
    )

    structure = parser.get_structure(
        "structure",
        pdb_file
    )

    masses = []
    coords = []

    for atom in structure.get_atoms():

        masses.append(
            atom.mass
        )

        coords.append(
            atom.coord
        )

    return (

        np.array(masses),

        np.array(coords)

    )

def estimate_diameter_from_pdb(
    pdb_file
):

    atom_coords = (
        get_atom_coordinates(
            pdb_file
        )
    )

    max_internal_distance = 0

    atom_coords_subsets = (
        np.array_split(
            atom_coords,
            10
        )
    )

    for subset in atom_coords_subsets:

        distances = cdist(

            subset,

            atom_coords,

            metric="euclidean"

        )

        local_max = np.max(
            distances
        )

        if local_max > (
            max_internal_distance
        ):

            max_internal_distance = (
                local_max
            )

    return float(
        max_internal_distance
    )

def get_particle_diameter(
    pdb_file
):

    diameter = (
        estimate_diameter_from_pdb(
            pdb_file
        )
    )

    return int(
        round(
            diameter
        )
    )


def get_radius_of_gyration(
    pdb_file
):

    masses, coords = (
        get_masses_and_coords(
            pdb_file
        )
    )

    total_mass = np.sum(
        masses
    )

    weighted_coords = (
        coords.T * masses
    )

    center_of_mass = (
        np.sum(
            weighted_coords,
            axis=1
        )
        /
        total_mass
    )

    squared_distances = np.sum(

        (coords - center_of_mass) ** 2,

        axis=1

    )

    rg = np.sqrt(

        np.sum(
            squared_distances * masses
        )
        /
        total_mass

    )

    return float(rg)


def get_extraction_diameter(
    pdb_file
):

    rg = get_radius_of_gyration(
        pdb_file
    )

    return int(
        round(
            2 * rg
        )
    )


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
