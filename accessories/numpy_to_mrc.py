import sys
import mrcfile
import numpy as np

in_fname = sys.argv[1]
out_fname = sys.argv[2]
invert = sys.argv[3] == "True"


np_array = np.load(in_fname)
if invert:
    np_array = np.where(np_array == 0, 1, 0)

mrcfile.new(out_fname, data=np_array.astype(np.float32), overwrite=True)
