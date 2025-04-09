import os
import sys
import glob
import ndjson


def main():
    path_to_annotations = sys.argv[1]
    file_str = sys.argv[2]

    pth = os.path.join(path_to_annotations, f"*{file_str}*")
    annotation_files = glob.glob(pth)

    coords = []
    for fname in annotation_files:
        with open(fname, "r") as in_annot_f:
            for ln in in_annot_f.readlines():
                x, y, z = ln.strip().split()[:3]
                coords.append(
                    {
                        "type": "orientedPoint",
                        "particle_name": "mononucleosome",
                        "location": {
                            "x": int(round(float(x), 0)),
                            "y": int(round(float(y), 0)),
                            "z": int(round(float(z), 0)),
                        },
                    }
                )

    out_fname = os.path.join(path_to_annotations, "all_annotations.ndjson")
    with open(out_fname, "w") as out_annot_f:
        ndjson.dump(coords, out_annot_f)
    print(out_fname)


if __name__ == "__main__":
    main()
