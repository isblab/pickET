import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt  # type:ignore


def get_label(fname):
    iseg_library = {"connected_component": "CC", "watershed": "WS", "meanshift": "MS"}
    split_fname = fname.split("/")
    feat_mode = split_fname[-3]
    cl_mode = split_fname[-2].split("_")[-1]
    iseg_mode = iseg_library[split_fname[-1].lstrip("overall_io_").rstrip(".yaml")]
    return f"{feat_mode}_{cl_mode}_{iseg_mode}"


def get_key_val(fname):
    iseg_methods = {
        "watershed": "WS",
        "connected_component": "CC",
    }  # , "meanshift": "MS"}
    clustering_methods = {"kmeans": "KM", "gmm": "GMM"}
    feature_mode = ["intensities", "ffts", "gabor"]
    feature_key, clustering_key, iseg_key = None, None, None
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

    xvals = []
    results = []
    random_results = []
    for idx, result_fname in enumerate(result_fnames):
        with open(result_fname, "r") as f:
            run_results = yaml.safe_load(f)
            metric_values = []
            random_metric_values = []
            for tomo, m in run_results.items():
                if tomo != "Global":
                    metric_values.append(m[metric])
                    random_metric_values.append(m[f"Random {metric}"])

        metric_values = np.array(metric_values)
        random_metric_values = np.array(random_metric_values)

        runname = "_".join(result_fname.split("/")[-3:])
        key_val = get_key_val(runname)

        results.append(metric_values)
        random_results.append(random_metric_values)
        xvals.append(key_val)

    results_vp = plt.violinplot(results, vert=False, showextrema=False)
    for body in results_vp["bodies"]:  # type:ignore
        body.set_facecolor("#00bfff")
        body.set_alpha(0.15)
    random_results_vp = plt.violinplot(random_results, vert=False, showextrema=False)
    for body in random_results_vp["bodies"]:  # type:ignore
        body.set_facecolor("#aeaeae")
        body.set_alpha(0.5)

    plt.boxplot(results, orientation="horizontal", showfliers=False)  # type:ignore

    plt.xlim(0, 1)
    plt.yticks(range(1, len(xvals) + 1), xvals)
    plt.title(f"{metric} comparison on {dataset_id}")
    plt.xlabel(metric)
    # plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{dataset_id}_{metric}_comparison.png", dpi=600)


if __name__ == "__main__":
    main()
