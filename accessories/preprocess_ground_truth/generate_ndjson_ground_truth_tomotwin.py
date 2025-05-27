import os
import sys
import glob
import ndjson
import numpy as np


def main():
    parent_path = sys.argv[1]
    tomo_shape = np.array(sys.argv[2:5], dtype=np.int32)

    tomo_data_paths = glob.glob(os.path.join(parent_path, "tomo_*"))

    offset_x, offset_y, offset_z = tomo_shape // 2

    for td_path in tomo_data_paths:
        path_to_coords_files = os.path.join(td_path, "coords")
        output_file = os.path.join(path_to_coords_files, "all_coords.ndjson")

        all_coords_files = glob.glob(os.path.join(path_to_coords_files, "*.txt"))
        print(f"Number of coords files: {len(all_coords_files)}")

        with open(output_file, "w") as outf:
            outputs = []
            for fname in all_coords_files:
                particle_id = fname.split("/")[-1]

                if (not particle_id.startswith("fiducial")) and (
                    not particle_id.startswith("vesicle")
                ):
                    particle_id = fname.split("/")[-1][:4]
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
    main()
