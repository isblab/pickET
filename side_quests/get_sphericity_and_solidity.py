import os
import glob
import yaml
import numpy as np
from rich.progress import track
from rich.console import Console
from skimage.measure import regionprops


def main():
    console = Console()
    particle_details_fname = "/home/shreyas/Projects/mining_tomograms/pickET/evaluations/tomotwin_particles.yaml"

    maps_dir = "/data2/shreyas/mining_tomograms/datasets/tomotwin/maps/"
    maps_paths = glob.glob(os.path.join(maps_dir, "*.npy"))

    with open(particle_details_fname, "r") as deetf:
        pdeets = yaml.safe_load(deetf)["particles"]
    console.log("Loaded the particle details")

    for idx, fpath in enumerate(track(maps_paths)):
        pdb_id = fpath.split("/")[-1][:-4].strip()
        if pdb_id not in pdeets:
            continue

        map = np.load(fpath)
        labeled_map = np.where(map == 0, 0, 1)
        properties = regionprops(labeled_map, map)[0]

        solidity = properties.solidity
        max_feret_diameter = properties.feret_diameter_max
        pdeets[pdb_id]["Solidity"] = solidity
        pdeets[pdb_id]["MaxFeretsDiameter"] = max_feret_diameter

        print(
            f"{idx+1}:\n\tPDB ID: {pdb_id}\n\tSolidity: {solidity}\n\tMaximum Feret's diameter: {max_feret_diameter}"
        )

    with open(particle_details_fname, "w") as out_deetf:
        yaml.dump(pdeets, out_deetf)


if __name__ == "__main__":
    main()
