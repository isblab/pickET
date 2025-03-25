import os
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pylab as plt
from Bio.PDB.MMCIFParser import MMCIFParser


def get_masses_and_coords(f_path: str) -> tuple[np.ndarray, np.ndarray]:
    parser = MMCIFParser()
    structure = parser.get_structure("protein", f_path)

    masses, coords = [], []
    for atom in structure.get_atoms():  # type:ignore
        masses.append(atom.mass)
        coords.append(atom.coord)
    return np.array(masses), np.array(coords)


def get_radius_of_gyration(masses: np.ndarray, coords: np.ndarray):
    total_struct_mass = np.sum(masses)
    weighted_coords = coords.T * masses
    center_of_mass = np.sum(weighted_coords, axis=1) / total_struct_mass

    squared_distance_from_com = np.sum((coords - center_of_mass) ** 2, axis=1)
    radius_of_gyration = np.sqrt(
        np.sum(squared_distance_from_com * masses) / total_struct_mass
    )
    return radius_of_gyration


def main():
    path_to_mmcif_files = "/data2/shreyas/mining_tomograms/datasets/tomotwin/cifs/"
    path_to_particles_file = "/home/shreyas/Projects/mining_tomograms/github/pickET/assessments/tomotwin_particles.yaml"
    with open(path_to_particles_file, "r") as input_f:
        particle_details = yaml.safe_load(input_f)

    x, y = [], []
    for pdb_id in tqdm(particle_details["particles"]):
        cif_fpath = os.path.join(path_to_mmcif_files, f"{pdb_id}.cif")
        atom_masses, atom_coords = get_masses_and_coords(cif_fpath)
        rg = get_radius_of_gyration(atom_masses, atom_coords)
        particle_details["particles"][pdb_id]["Rg"] = round(float(rg), 4)

        x.append(particle_details["particles"][pdb_id]["MW_pdb"])
        y.append(rg)
    plt.scatter(x, y, s=3, alpha=0.75)
    plt.show()

    with open(path_to_particles_file, "w") as out_f:
        yaml.dump(particle_details, out_f)


if __name__ == "__main__":
    main()
