import numpy as np
from scipy.ndimage import gaussian_filter

from picket.core import preprocessing


def test_get_z_section():
    z, y, x = 5, 10, 10
    tomogram = []

    for i in range(z):
        tomogram.append(np.expand_dims((np.ones((y, x)) * i), axis=0))
    tomogram = np.concatenate(tomogram, axis=0)
    t1 = preprocessing.get_z_section(tomogram, None, None)
    t2 = preprocessing.get_z_section(tomogram, 2, None)
    t3 = preprocessing.get_z_section(tomogram, None, 2)
    t4 = preprocessing.get_z_section(tomogram, 2, 4)

    assert isinstance(t1, np.ndarray)
    assert isinstance(t2, np.ndarray)
    assert isinstance(t3, np.ndarray)
    assert isinstance(t4, np.ndarray)
    assert t1.shape == tomogram.shape
    assert t2.shape == (3, y, x)
    assert t3.shape == (2, y, x)
    assert t4.shape == (2, y, x)


def test_guassian_blur_tomogram():
    sigma_val = 2
    tomogram = np.random.uniform(0, 1, (30, 30, 30)).astype(np.float32)

    gb_ref = gaussian_filter(tomogram, sigma_val)
    gb_test = preprocessing.guassian_blur_tomogram(tomogram, sigma_val)

    assert np.allclose(gb_test, gb_ref)


def test_minmax_normalize_array():
    tomogram = np.random.uniform(0, 1, (30, 30, 30)).astype(np.float32)
    minmax_normalized = preprocessing.minmax_normalize_array(tomogram)
    assert np.min(minmax_normalized) == 0
    assert np.max(minmax_normalized) == 1
