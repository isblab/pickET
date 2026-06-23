import os
import numpy as np

from scipy.spatial.distance import cdist

from Bio.PDB import PDBParser

def get_atom_coordinates(
    pdb_file
):

    parser = PDBParser()

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

if __name__ == "__main__":
    main()
