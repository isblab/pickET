import numpy as np
from assets import utils


def test_load_params_from_yaml():
    fpath = "example/s1_params_example.yaml"
    params = utils.load_params_from_yaml(fpath)
    assert isinstance(params, dict)
    assert params.get("dataset_name", False)
    assert isinstance(params.get("inputs"), list)
    assert isinstance(params.get("feature_extraction_params"), list)
    assert isinstance(params.get("clustering_methods"), list)


def test_load_tomogram():
    fpath = "tests/tomogram_testcase.mrc"
    tomogram, voxel_sizes = utils.load_tomogram(fpath)
    assert isinstance(tomogram, np.ndarray)
    assert isinstance(voxel_sizes, np.ndarray)
    assert voxel_sizes.dtype == np.float32


def test_prepare_out_coords():
    coords = np.random.randint(0, 100, size=(100, 3))
    metadata = {
        "time_taken": 1000.123,
        "voxel_sizes": np.array([10.2, 10.2, 10.2]),
        "mean": np.mean([1, 2, 3, 4, 5, 5, 5]),
        "none_val": None,
    }

    out_coords = utils.prepare_out_coords(coords, metadata)
    assert isinstance(out_coords, dict)
    assert isinstance(out_coords["Predicted_Particle_Centroid_Coordinates"], list)
    assert isinstance(out_coords["Predicted_Particle_Centroid_Coordinates"][0], dict)
    assert isinstance(out_coords["metadata"], dict)
    assert isinstance(out_coords["metadata"]["time_taken"], float)
    assert isinstance(out_coords["metadata"]["voxel_sizes"], list)
    assert isinstance(out_coords["metadata"]["mean"], float)
    assert isinstance(out_coords["metadata"]["none_val"], str)


def test_get_neighborhoods():
    window_size = 5
    tomogram = np.random.uniform(0, 1, size=(50, 50, 50))
    max_n_windows = 100

    neighborhoods1, preshape1, max_num_neighborhoods_for_fitting1 = (
        utils.get_neighborhoods(tomogram, window_size)
    )
    neighborhoods_w_max, preshape_w_max, max_num_neighborhoods_for_fitting_w_max = (
        utils.get_neighborhoods(tomogram, window_size, max_n_windows)
    )

    assert isinstance(neighborhoods1, np.ndarray)
    assert neighborhoods1.shape == (
        (tomogram.shape[0] - 2 * (window_size // 2))
        * (tomogram.shape[1] - 2 * (window_size // 2))
        * (tomogram.shape[2] - 2 * (window_size // 2)),
        window_size,
        window_size,
        window_size,
    )
    assert isinstance(preshape1, tuple)
    assert max_num_neighborhoods_for_fitting1 is None

    assert isinstance(neighborhoods_w_max, np.ndarray)
    assert neighborhoods_w_max.shape == (
        max_n_windows,
        window_size,
        window_size,
        window_size,
    )
    assert isinstance(preshape_w_max, tuple)
    assert max_num_neighborhoods_for_fitting_w_max == max_n_windows


def test_subsample_neighborhoods():
    window_size = 5
    max_n_windows = 100

    neighborhoods = np.random.uniform(
        0, 1, (1000, window_size, window_size, window_size)
    )
    subsampled_neighborhoods = utils.subsample_neighborhoods(
        neighborhoods, max_n_windows
    )

    assert isinstance(subsampled_neighborhoods, np.ndarray)
    assert subsampled_neighborhoods.shape == (
        max_n_windows,
        window_size,
        window_size,
        window_size,
    )


def test_read_yaml_coords():
    coords_path = "tests/out_coords_testcase.yaml"
    coords = utils.read_yaml_coords(coords_path)
    assert isinstance(coords, np.ndarray)
    assert coords.shape == (100, 3)
