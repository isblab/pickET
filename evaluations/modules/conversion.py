import yaml
import starfile


def convert_star_to_yaml(

    input_star,

    output_yaml,

    particle_name,

    tomogram_shape,

    tomogram_path,

    voxel_size

):

    # =====================================================
    # LOAD STAR FILE
    # =====================================================

    df = starfile.read(
        input_star
    )

    # =====================================================
    # PREPARE OUTPUT STRUCTURE
    # =====================================================

    out_dict = {}

    out_dict["metadata"] = {

        "source": "PyTOM",

        "particle_name": particle_name,

        "tomogram_shape": list(tomogram_shape),

        "tomogram_path": tomogram_path,

        "voxel_size": round(float(voxel_size), 4)

    }

    out_dict[
        "Predicted_Particle_Centroid_Coordinates"
    ] = []

    # =====================================================
    # CONVERT COORDINATES
    # =====================================================

    for _, row in df.iterrows():

        x = int(
            round(
                float(
                    row["rlnCoordinateX"]
                ),
                0
            )
        )

        y = int(
            round(
                float(
                    row["rlnCoordinateY"]
                ),
                0
            )
        )

        z = int(
            round(
                float(
                    row["rlnCoordinateZ"]
                ),
                0
            )
        )

        out_dict[
            "Predicted_Particle_Centroid_Coordinates"
        ].append({

            "x": x,

            "y": y,

            "z": z

        })

    # =====================================================
    # WRITE YAML
    # =====================================================

    with open(
        output_yaml,
        "w"
    ) as outf:

        yaml.dump(
            out_dict,
            outf,
            sort_keys=False
        )

    print(
        f"\nSaved prediction YAML:"
    )

    print(
        output_yaml
    )

    print(
        f"\nNumber of particles converted:"
    )

    print(
        len(
            out_dict[
                "Predicted_Particle_Centroid_Coordinates"
            ]
        )
    )


def run_conversion(

    input_star,

    output_yaml,

    particle_name,

    tomogram_shape,

    tomogram_path,

    voxel_size

):

    print(
        "\nRunning Conversion:\n"
    )

    print(
        f"Input STAR: {input_star}"
    )

    print(
        f"Output YAML: {output_yaml}"
    )

    convert_star_to_yaml(

        input_star,

        output_yaml,

        particle_name,

        tomogram_shape,

        tomogram_path,

        voxel_size

    )
