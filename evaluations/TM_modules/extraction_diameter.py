import numpy as np

from Bio.PDB import PDBParser


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

    center_of_mass = (

        np.sum(
            coords.T * masses,
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

    return float(
        rg
    )


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


def main():

    pdb_file = input(
        "Enter PDB file path: "
    ).strip()

    rg = get_radius_of_gyration(
        pdb_file
    )

    extraction_diameter = (
        get_extraction_diameter(
            pdb_file
        )
    )

    print(
        f"\nRadius of gyration: "
        f"{rg:.2f} Å"
    )

    print(
        f"Extraction diameter: "
        f"{extraction_diameter} Å"
    )


if __name__ == "__main__":
    main()
