import sys
import ndjson
import starfile


def main():

    # =====================================================
    # INPUTS
    # =====================================================

    input_star = sys.argv[1]

    output_ndjson = sys.argv[2]

    particle_name = sys.argv[3]

    # =====================================================
    # LOAD STAR FILE
    # =====================================================

    df = starfile.read(input_star)

    # =====================================================
    # CONVERT TO NDJSON FORMAT
    # =====================================================

    outputs = []

    for _, row in df.iterrows():

        x = int(round(float(row["rlnCoordinateX"]), 0))

        y = int(round(float(row["rlnCoordinateY"]), 0))

        z = int(round(float(row["rlnCoordinateZ"]), 0))

        outln = {
            "type": "orientedPoint",
            "particle_name": particle_name,
            "location": {
                "x": x,
                "y": y,
                "z": z,
            },
        }

        outputs.append(outln)

    # =====================================================
    # WRITE NDJSON
    # =====================================================

    with open(output_ndjson, "w") as outf:

        ndjson.dump(outputs, outf)

    print(f"\nSaved NDJSON file:")
    print(output_ndjson)

    print(f"\nNumber of particles converted:")
    print(len(outputs))


if __name__ == "__main__":

    main()
