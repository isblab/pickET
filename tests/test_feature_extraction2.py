import numpy as np
from rich.progress import track


def main():
    path1 = "/data2/shreyas/mining_tomograms/working/tests/first_pass_features.npy"
    path2 = "/data2/shreyas/mining_tomograms/working/tests/second_pass_features.npy"

    arr1 = np.load(path1).reshape((196, 508, 508, 125))
    arr2 = np.load(path2).reshape((196, 508, 508, 125))

    dissimilar_mask = np.zeros((196, 508, 508))
    for z_idx, zslice in enumerate(track(arr1)):
        for y_idx, y in enumerate(zslice):
            for x_idx in range(len(y)):
                o = arr1[z_idx, y_idx, x_idx]
                n = arr2[z_idx, y_idx, x_idx]
                if not np.allclose(o, n, atol=1e-4):
                    dissimilar_mask[z_idx, y_idx, x_idx] = np.mean(o - n)

    np.save("dissimilar_mask.npy", dissimilar_mask)


if __name__ == "__main__":
    main()
