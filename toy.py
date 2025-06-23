import mrcfile
import numpy as np


def main():
    arr = np.random.uniform(0, 1, (10, 10, 10)).astype(np.float32)
    mrcfile.new("tests/test_tomogram.mrc", data=arr, overwrite=True)


if __name__ == "__main__":
    main()
