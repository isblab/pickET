import os
import glob
import ndjson
import argparse
import mrcfile
import numpy as np

def convert_annotations(annotation_folder, output_folder, annotation_suffix):

    os.makedirs(output_folder, exist_ok=True)

    pattern = os.path.join(annotation_folder, f"*_{annotation_suffix}.csv")

    annotation_files = sorted(glob.glob(pattern))

    print(f"Found {len(annotation_files)} annotation file(s)")

    for fname in annotation_files:

        basename = os.path.basename(fname).replace(f"_{annotation_suffix}.csv", "")
        out_fname = os.path.join(output_folder, f"{basename}.ndjson")

        if os.path.exists(out_fname):
            print(f"Skipping existing file: {out_fname}")
            continue

        coords = []

        with open(fname, "r") as in_f:

            for ln in in_f.readlines():

                line = ln.strip()

                if not line:
                    continue

                x, y, z = (ln.split(",")[:3])

                coords.append(
                    {
                        "type":
                            "orientedPoint",

                        "location":
                            {
                                "x":int(round(float(x))),
                                "y":int(round(float(y))),
                                "z":int(round(float(z)))
                            }
                    }
                )

        with open(out_fname, "w") as out_f:

            ndjson.dump(coords, out_f)

        print(f"Saved: {out_fname}")

def convert_tomotwin_annotation(
    tomotwin_sim_round_path,
    output_folder: None,
    particles=["all"],
    ignore_fiducial=True,
    ignore_vesicle=True
):

    tomo_data_paths = glob.glob(os.path.join(tomotwin_sim_round_path, "tomo_*"), recursive=True)

    offset_x, offset_y, offset_z = None, None, None

    for td_path in tomo_data_paths:

        # assumes all tomotwin tomograms in a round are of same shape
        if offset_x is None:
            mrcpath = os.path.join(td_path, "tiltseries_rec.mrc")
            with mrcfile.open(mrcpath) as mf:
                tomo_shape = np.array(mf.data.shape, dtype=np.int32)

            offset_z, offset_y, offset_x = tomo_shape // 2 #! note the order

        assert all(o is not None for o in (offset_x, offset_y, offset_z))

        path_to_coords_files = os.path.join(td_path, "coords")

        if output_folder is None:

            output_file = os.path.join(path_to_coords_files, "all_coords.ndjson")

        elif isinstance(output_folder, str):

            basename = os.path.basename(td_path)
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(output_folder, f"{basename}.ndjson")

        all_coords_files = glob.glob(os.path.join(path_to_coords_files, "*.txt"))
        print(f"Number of coords files: {len(all_coords_files)}")

        _particles = [f.split("_")[0] for f in os.listdir(path_to_coords_files)]
        if particles[0] == "all":
            particles = _particles
        if ignore_fiducial and "fiducial" in particles:
            particles.remove("fiducial")
        if ignore_vesicle and "vesicle" in particles:
            particles.remove("vesicle")

        with open(output_file, "w") as outf:
            outputs = []
            for fname in all_coords_files:
                particle_id = os.path.basename(fname).split("_")[0]

                if particle_id in particles:

                    with open(fname, "r") as f1:
                        for line in f1.readlines():
                            if not line.startswith("#") and not line.startswith(" "):
                                y, x, z = line.split()[:3]
                                x, y, z = (
                                    int(512 - (int(float(x)) + offset_x) - 1),
                                    int(float(y) + offset_y - 1),
                                    int(float(z) + offset_z - 1),
                                )

                                outln = {
                                    "type": "orientedPoint",
                                    "particle_id": particle_id,
                                    "location": {"x": x, "y": y, "z": z},
                                }
                                outputs.append(outln)
            ndjson.dump(outputs, outf)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tomotwin_sim_round_path",
        required=True,
        help="Path to tomotwin dataset (e.g. /path/to/tomo_simulation_round_1)"
    )

    parser.add_argument(
        "--outdir",
        required=False,
        help="Output director to save ndjson files."
    )

    parser.add_argument(
        "--particles",
        default="all",
        help="List of allowed particles in ground truth annotation (e.g., '6ahu,4wrm')",
    )

    args = parser.parse_args()

    particles = args.particles.split(",")
    tomotwin_sim_round_path = args.tomotwin_sim_round_path
    output_folder = args.outdir

    convert_tomotwin_annotation(
        tomotwin_sim_round_path=tomotwin_sim_round_path,
        output_folder=output_folder,
        particles=particles,
        ignore_vesicle=True,
        ignore_fiducial=True,
    )
