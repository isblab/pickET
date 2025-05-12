import h5py
import numpy as np

from assets import utils, preprocessing


# tomo_fname = "/data2/shreyas/mining_tomograms/datasets/tomotwin/tomo_simulation_round_1/tomo_01.2022-04-11T140327+0200/denoised_tiltseries_rec.mrc"
# tomo, voxel_sizes = utils.load_tomogram(tomo_fname)

# outfname = "/home/shreyas/Projects/mining_tomograms/pickET/test.h5"
# with h5py.File(outfname, "w") as f:
#     group = f.create_group("tomograms")
#     group.attrs["voxel_sizes"] = voxel_sizes
#     group.attrs["tomogram_path"] = tomo_fname

#     dset1 = group.create_dataset("tomogram_1", data=tomo)
#     dset1.attrs["description"] = "This is a test dataset 1"

#     dset2 = group.create_dataset("tomogram_2", data=tomo)
#     dset2.attrs["description"] = "This is a test dataset 2"


with h5py.File(
    "/home/shreyas/Projects/mining_tomograms/pickET/test.h5",
    "r",
) as h5f:
    group_metadata = h5f["tomograms"].attrs
    t1 = np.array(h5f["tomograms"]["tomogram_1"][:])
    t2 = np.array(h5f["tomograms"]["tomogram_2"][:])

    if "tomogram_1" in h5f["tomograms"].keys():
        print("Yay")

    # print()
    # for k, v in group_metadata.items():
    #     print(k, v)

    # print()
    # print(t1.shape, t2.shape)

#     dataset = h5f["segmentation"]
#     for k in dataset.attrs:
#         print(k)
# print(dset.attrs["description"])
# print(dset.attrs["voxel_sizes"])
# print(dset.attrs["tomogram_path"])
