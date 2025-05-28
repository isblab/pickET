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

    color_palette = {
        "intensities_KM_CC": "#399424",
        "intensities_KM_WS": "#399424",
        "intensities_GMM_CC": "#399424",
        "intensities_GMM_WS": "#399424",
        "ffts_KM_CC": "#E76600",
        "ffts_KM_WS": "#E76600",
        "ffts_GMM_CC": "#E76600",
        "ffts_GMM_WS": "#E76600",
        "gabor_KM_CC": "#9945AC",
        "gabor_KM_WS": "#9945AC",
        "gabor_GMM_CC": "#9945AC",
        "gabor_GMM_WS": "#9945AC",
    }

    workflow_names = {
        "intensities_KM_CC": "Intensities KMeans CC",
        "intensities_KM_WS": "Intensities KMeans WS",
        "intensities_GMM_CC": "Intensities GMM CC",
        "intensities_GMM_WS": "Intensities GMM WS",
        "ffts_KM_CC": "FFTs KMeans CC",
        "ffts_KM_WS": "FFTs KMeans WS",
        "ffts_GMM_CC": "FFTs GMM CC",
        "ffts_GMM_WS": "FFTs GMM WS",
        "gabor_KM_CC": "Gabor KMeans CC",
        "gabor_KM_WS": "Gabor KMeans WS",
        "gabor_GMM_CC": "Gabor GMM CC",
        "gabor_GMM_WS": "Gabor GMM WS",
    }
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
                    if metric in ("Precision", "Recall", "F1-score"):
                        metric_values.append(m[metric])
                        random_metric_values.append(m[f"Random {metric}"])
                    elif metric == "Total time taken":
                        metric_values.append(m[metric] / 60)
                    elif metric == "Relative recall":
                        recall = m["Recall"]
                        random_recall = m["Random Recall"]
                        relative_recall = float(np.sqrt(recall * (1 - random_recall)))
                        print(m["Recall"], m["Random Recall"], relative_recall)
                        metric_values.append(relative_recall)

        metric_values = np.array(metric_values)
        random_metric_values = np.array(random_metric_values)

        runname = "_".join(result_fname.split("/")[-3:])
        key_val = get_key_val(runname)

        results.append(metric_values)
        random_results.append(random_metric_values)
        xvals.append(key_val)

    results_vp = plt.violinplot(results, vert=False, showextrema=False)
    for idx, body in enumerate(results_vp["bodies"]):  # type:ignore
        body.set_facecolor(color_palette[xvals[idx]])
        if xvals[idx].split("_")[-2] == "KM":
            body.set_alpha(0.4)
        elif xvals[idx].split("_")[-2] == "GMM":
            body.set_alpha(0.2)

    if metric in ("Precision", "Recall", "F1-score"):
        random_results_vp = plt.violinplot(
            random_results, vert=False, showextrema=False
        )
        for body in random_results_vp["bodies"]:  # type:ignore
            body.set_facecolor("#aeaeae")
            body.set_alpha(0.6)

    bxplt = plt.boxplot(
        results, orientation="horizontal", showfliers=False
    )  # type:ignore
    for mdn in bxplt["medians"]:
        mdn.set(color="#000000")

    xlabels = [workflow_names[k] for k in xvals]

    if metric in ("Precision", "Recall", "F1-score", "Relative recall"):
        plt.xlim(0, 1)

    plt.yticks(range(1, len(xlabels) + 1), xlabels)
    plt.title(f"{metric} comparison on {dataset_id}")
    if metric == "Total time taken":
        plt.xlabel("Total time (min)")
    else:
        plt.xlabel(metric)
    # plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{dataset_id}_{metric}_comparison.png", dpi=600)


if __name__ == "__main__":
    main()
