import os
import glob
import numpy as np


def load_cif_file(fpath: str):
    loaded_models = run(SESSION, f"open {fpath}")
    return loaded_models


def generate_map(
    model_id, resolution: float, grid_spacing: int = 1
) -> tuple[np.ndarray, int]:
    molmap_command = f"molmap #{model_id} {resolution} gridSpacing {grid_spacing}"
    map = run(SESSION, molmap_command)
    map_id = map.id[0]
    map = map.full_matrix()
    return map, map_id


def main():
    resolution = 10.20
    path_to_cifs = "/data2/shreyas/mining_tomograms/datasets/tomotwin/cifs/"
    output_path = f"/data2/shreyas/mining_tomograms/datasets/tomotwin/maps/"
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    cif_fpaths = glob.glob(os.path.join(path_to_cifs, f"*.cif"))
    print(f"Will process {len(cif_fpaths)} CIF files\n")

    for idx, fpath in enumerate(cif_fpaths):
        pdb_id = fpath.split("/")[-1][:-4]

        models = load_cif_file(fpath)
        model_id = models[0].id[0]
        map, map_id = generate_map(model_id, resolution)

        out_fname = os.path.join(output_path, f"{pdb_id}.npy")
        np.save(out_fname, map)
        run(SESSION, f"close #{model_id} #{map_id}")
        print(f"Processed {idx+1} files out of {len(cif_fpaths)}\n{'-'*50}\n\n")


if __name__.startswith("ChimeraX_sandbox_"):
    from chimerax.core.commands import run  # type:ignore

    SESSION = session  # type:ignore
    print(f"Launched {__name__}")
    main()
else:
    raise RuntimeError(
        "This script must be run from within the ChimeraX window or run it as `chimerax --nogui --script side_quests/get_solidity_and_sphericity.py`"
    )
