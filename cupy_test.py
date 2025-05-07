import cupy as cp
import numpy as np
from assets import preprocessing

a = np.random.random((1020, 1020, 508))
b = cp.asarray(a)
print(type(a), type(b))

c = preprocessing.guassian_blur_tomogram(a)
print(type(c))
