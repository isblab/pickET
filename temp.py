import mrcfile

# # Open the .mrc file
# with mrcfile.open(
#     # "/data2/shreyas/mining_tomograms/datasets/lugan/20221029_03/20221029_03_denoised.mrc",
#     "/data2/shreyas/mining_tomograms/datasets/tomotwin/tomo_simulation_round_1/tomo_01.2022-04-11T140327+0200/coords/occupancy.mrc",
#     permissive=True,
# ) as mrc:
#     # print(mrc.print_header())
#     # Inspect the header
#     # print("nx:", mrc.header.nx)
#     # print("nxstart:", mrc.header.nxstart)
#     # print("nystart:", mrc.header.nystart)
#     # print("nzstart:", mrc.header.nzstart)
#     print("X origin:", mrc.header.origin.x)
#     print("Y origin:", mrc.header.origin.y)
#     print("Z origin:", mrc.header.origin.z)
#     print("Dimensions:", mrc.header.nx, mrc.header.ny, mrc.header.nz)

#     # Check inversion manually
#     # print("Data type:", mrc.data.dtype)
#     # print("Data shape:", mrc.data.shape)


# # Analyze the origin values or header offsets

import mrcfile
import numpy as np


def detect_inverted_axis(mrc_file):
    with mrcfile.open(mrc_file, permissive=True) as mrc:
        data = mrc.data

        # Check inversion along the X-axis
        x_inverted = np.array_equal(data, np.flip(data, axis=1))

        # Check inversion along the Y-axis
        y_inverted = np.array_equal(data, np.flip(data, axis=2))

        # Check inversion along the Z-axis
        z_inverted = np.array_equal(data, np.flip(data, axis=0))

        # Return which axis is inverted
        return {
            "X Inverted": x_inverted,
            "Y Inverted": y_inverted,
            "Z Inverted": z_inverted,
        }


# Example usage
inverted_axes = detect_inverted_axis(
    "/data2/shreyas/mining_tomograms/datasets/tomotwin/tomo_simulation_round_1/tomo_01.2022-04-11T140327+0200/coords/occupancy.mrc"
)
print("Axis of inversion:")
print(inverted_axes)
