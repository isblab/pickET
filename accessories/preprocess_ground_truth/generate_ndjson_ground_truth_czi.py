import os
import sys
import glob
import ndjson


def main():
    parent_path = sys.argv[1]

    coords_f_paths = glob.glob(os.path.join(parent_path, "*ndjson"))
    out_fname = os.path.join(parent_path, "all_annotations.ndjson")

    output_annotations = []
    for fpath in coords_f_paths:
        particle_name = fpath.split("/")[-1].split(".")[0]
        if not "membrane" in particle_name:
            with open(fpath, "r") as in_annot_f:
                annotations = ndjson.load(in_annot_f)
                for annot in annotations:
                    annot["particle_name"] = particle_name
                    output_annotations.append(annot)

    with open(out_fname, "w") as outf:
        ndjson.dump(output_annotations, outf)


if __name__ == "__main__":
    main()
