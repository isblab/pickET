import h5py
import numpy as np

from assets import utils, preprocessing

# tomo_fname = "/home/shreyas/pickET/test_tomos/10301.mrc"
# tomo,voxel_sizes = preprocessing.load_tomogram(tomo_fname)

# outfname = "/home/shreyas/pickET/teest.h5"
# with h5py.File(outfname, "w") as f:
#     # Create a dataset in the file
#     dset = f.create_dataset("tomogram", data=tomo)

#     # Optionally, you can add attributes to the dataset
#     dset.attrs["description"] = "This is a test dataset"
#     dset.attrs["voxel_sizes"] = voxel_sizes
#     dset.attrs["tomogram_path"] = tomo_fname

with h5py.File("/data/shreyas/mining_tomograms/s1_clean_results_picket_v2/h5py_test/segmentation_0_ffts_kmeans", "r") as h5f:
    dataset = h5f['segmentation']
    for k in dataset.attrs:
        print(k)
    # # Access the dataset
    # dset = f["tomogram"]

    # # Read the data
    
    # print(dtst.shape)
    # print(dset.attrs["description"])
    # print(dset.attrs["voxel_sizes"])
    # print(dset.attrs["tomogram_path"])