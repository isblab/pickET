import os
import ast
import argparse
import numpy as np

from assets import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--in_fname",
        type=str,
        required=True,
        help="Path to predicted coordinates .yaml file",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        required=True,
        help="Path to directory where the output .csv file should be placed",
    )
    parser.add_argument(
        "-n",
        "--new_origin",
        type=str,
        required=True,
        help="Coordinates to the new origin. \
            The input file assumes the origin to be at the top left front corner of the tomogram. \
                Pass the argument as '(z,y,x)' with quotes.",
    )
    return parser.parse_args()


def get_new_origin(new_origin: str) -> np.ndarray:
    new_origin = new_origin[1:-1]
    out_new_origin = ast.literal_eval(new_origin)
    if not isinstance(out_new_origin, tuple):
        raise ValueError("Could not parsing the new origin coordinates")
    if (
        (not isinstance(out_new_origin[0], int))
        or (not isinstance(out_new_origin[1], int))
        or (not isinstance(out_new_origin[2], int))
    ):
        raise ValueError("The new origin coordinates must be integers")

    return np.array(out_new_origin)


def main():
    args = parse_args()
    new_origin = get_new_origin(args.new_origin)

    ori_coords = utils.read_yaml_coords(args.in_fname)
    offset_corrected_coords = (ori_coords - new_origin).astype(np.int32)

    out_fname = args.in_fname.split("/")[-1]
    out_fname = f"{os.path.join(args.out_dir, out_fname[:-4])}csv"
    print(out_fname)
    print(offset_corrected_coords)

    with open(out_fname, "w") as outf:
        outf.write("x,y,z\n")
        for c in offset_corrected_coords:
            outf.write(f"{c[2]},{c[1]},{c[0]}\n")


if __name__ == "__main__":
    main()
