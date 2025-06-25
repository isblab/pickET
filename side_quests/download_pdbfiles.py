import os
import sys
import yaml
import requests
from rich.progress import track


def get_pdb_file(pdb_id: str, download_path: str) -> None:
    pdb_id_u = pdb_id.upper()
    URL = f"https://files.rcsb.org/download/{pdb_id_u}.pdb"

    req = requests.get(URL)
    if req.status_code == 200:
        pdb_data = req.text
        with open(os.path.join(download_path, f"{pdb_id}.pdb"), "w") as outf:
            outf.write(pdb_data)
    else:
        print(f"Error retrieving PDB ID {pdb_id}")


def main():
    particle_details_fname = sys.argv[1]
    out_dir = "/data2/shreyas/mining_tomograms/datasets/tomotwin/pdbs/"
    with open(particle_details_fname, "r") as pdeet_f:
        particle_details = yaml.safe_load(pdeet_f)["particles"]

    for particle_id in track(particle_details):
        get_pdb_file(particle_id, out_dir)


if __name__ == "__main__":
    main()
