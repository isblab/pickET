import os
import glob
import ndjson

def convert_annotations(
    annotation_folder,
    output_folder,
    annotation_suffix
):

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    pattern = os.path.join(
        annotation_folder,
        f"*_{annotation_suffix}.csv"
    )

    annotation_files = sorted(
        glob.glob(pattern)
    )

    print(
        f"Found {len(annotation_files)} "
        f"annotation file(s)"
    )

    for fname in annotation_files:

        basename = (
            os.path.basename(fname)
            .replace(
                f"_{annotation_suffix}.csv",
                ""
            )
        )

        out_fname = os.path.join(
            output_folder,
            f"{basename}.ndjson"
        )

        if os.path.exists(out_fname):

            print(
                f"Skipping existing file: "
                f"{out_fname}"
            )

            continue

        coords = []

        with open(fname, "r") as in_f:

            for ln in in_f.readlines():

                line = ln.strip()

                if not line:
                    continue

                x, y, z = (
                    ln.split(",")[:3]
                )

                coords.append(
                    {
                        "type":
                            "orientedPoint",

                        "location":
                            {
                                "x":
                                    int(
                                        round(
                                            float(x)
                                        )
                                    ),

                                "y":
                                    int(
                                        round(
                                            float(y)
                                        )
                                    ),

                                "z":
                                    int(
                                        round(
                                            float(z)
                                        )
                                    )
                            }
                    }
                )

        with open(
            out_fname,
            "w"
        ) as out_f:

            ndjson.dump(
                coords,
                out_f
            )

        print(
            f"Saved: {out_fname}"
        )

