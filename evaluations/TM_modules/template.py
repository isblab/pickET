import subprocess
import mrcfile
import numpy as np
from Bio.PDB import PDBParser
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

def get_template_voxel_size(template_path: str) -> float:

    with mrcfile.open(template_path, permissive=True) as mrc:

        voxel_size = float(mrc.voxel_size.x)

    return voxel_size


def get_ca_coordinates(pdb_file):

    parser = PDBParser()

    structure = parser.get_structure("structure", pdb_file)

    coordinates = []

    for model in structure:

        for chain in model:

            for residue in chain:

                coordinates.extend([
                    atom.get_coord() for atom in residue
                    if atom.get_name() == "CA"
                ])

    return np.array(coordinates)


def estimate_diameter_from_pdb(pdb_file):

    ca_coordinates = get_ca_coordinates(pdb_file)
    hull = ConvexHull(ca_coordinates)
    hull_vertices = ca_coordinates[hull.vertices]
    pairwise_distances = pdist(hull_vertices) + 15 # accounting for side-chains

    return np.max(pairwise_distances)


def get_particle_diameter(config):

    diameter = config["particle"]["template_matching_diameter_angstrom"]
    pdb_file = config["particle"]["pdb"]

    if diameter is not None:
        print("\nUsing user-provided particle diameter.")

    elif pdb_file is not None:
        print("\nEstimating particle diameter from PDB...")
        diameter = estimate_diameter_from_pdb(pdb_file)
        print(f"Estimated diameter: {diameter} A")

    else:
        raise ValueError(
            "Either template_matching_diameter_angstrom "
            "or pdb must be provided."
        )

    return int(round(diameter))


def generate_template(
    template_inpath: str,
    template_outpath: str,
    output_voxel_size: float,
    box_size: int,
    invert: bool,
):

    with mrcfile.open(template_inpath, permissive=True) as mrc:

        input_voxel_size = float(mrc.voxel_size.x)

    cmd = [

        "pytom_create_template.py",

        "-i",
        template_inpath,

        "-o",
        template_outpath,

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

    subprocess.run(cmd, check=True)


def generate_mask(box_size, radius, output_mask):

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

    subprocess.run(cmd, check=True)
