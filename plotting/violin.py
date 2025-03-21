import sys
import yaml
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


def generate_discrete_color_palette(n):
    # Generate a palette of `n` distinct colors using tab20 or any other qualitative colormap
    base_cmap = plt.get_cmap("tab20", n)  # tab20 is ideal for discrete categories
    colors = base_cmap(np.arange(n))
    return ListedColormap(colors)


def get_label(fname):
    iseg_library = {"connected_component": "CC", "watershed": "WS", "meanshift": "MS"}
    split_fname = fname.split("/")
    feat_mode = split_fname[-3]
    cl_mode = split_fname[-2].split("_")[-1]
    iseg_mode = iseg_library[split_fname[-1].lstrip("overall_io_").rstrip(".yaml")]
    return f"{feat_mode}_{cl_mode}_{iseg_mode}"


def get_key_val(fname):
    iseg_methods = {"watershed": "WS", "connected_component": "CC", "meanshift": "MS"}
    clustering_methods = {"kmeans": "KM", "gmm": "GMM"}
    feature_mode = ["intensities", "ffts", "gabor"]
    for i, met in iseg_methods.items():
        if i in fname:
            iseg_key = met
            break
    for i, met in clustering_methods.items():
        if i in fname:
            clustering_key = met
            break
    for i in feature_mode:
        if i in fname:
            feature_key = i
            break
    return f"{feature_key}_{clustering_key}_{iseg_key}"


def main():
    metric = sys.argv[1]
    dataset_id = sys.argv[2]
    output_dir = sys.argv[3]
    result_fnames = sys.argv[4:]

    color_palette = generate_discrete_color_palette(len(result_fnames))

    xvals = []
    results = []
    for idx, result_fname in enumerate(result_fnames):
        with open(result_fname, "r") as f:
            run_results = yaml.safe_load(f)
            metric_values = []
            for tomo, m in run_results.items():
                if tomo != "Global":
                    metric_values.append(m[metric])

        metric_values = np.array(metric_values)

        runname = "_".join(result_fname.split("/")[-3:])
        key_val = get_key_val(runname)

        results.append(metric_values)
        xvals.append(key_val)

    plt.violinplot(
        results,
        vert=False,
        showmeans=True,
        # label=get_label(result_fname),
        # color=color_palette(idx),
    )

    plt.xlim(0, 1)
    plt.yticks(range(1, len(xvals) + 1), xvals)
    plt.title(f"{metric} comparison on {dataset_id}")
    plt.xlabel("metric")
    # plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"output_dir/{dataset_id}_{metric}_comparison.png", dpi=600)


if __name__ == "__main__":
    main()
