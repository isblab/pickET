import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

COLORS = {
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

WORKFLOWS = {
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


def load_run_results(result_fname: str) -> dict:
    with open(result_fname, "r") as f:
        results = yaml.safe_load(f)
    return results


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


def get_significance_mark(pvalue: float) -> str:
    mark = ""
    if pvalue > 0.05:
        mark = "n.s."
    elif 0.01 < pvalue <= 0.05:
        mark = "*"
    elif 0.001 < pvalue <= 0.01:
        mark = "**"
    elif pvalue <= 0.001:
        mark = "***"
    else:
        raise ValueError(
            f"Encountered {pvalue} as p-value.\nUnexpected error occured in generating the significance matrix"
        )
    return mark


METRICS = ["Precision", "Recall", "F1-score", "Relative recall", "Total time taken"]


def main():
    dataset_id = sys.argv[1]
    output_dir = sys.argv[2]
    result_fnames = [sys.argv[3], sys.argv[4]]

    for metric in METRICS:
        workflow_keys = []
        workflow_names = []
        metric_values, gtr_metric_values = [], []
        for result_fname in result_fnames:
            wk = get_key_val(result_fname)
            wn = WORKFLOWS[wk]
            workflow_keys.append(wk)
            workflow_names.append(wn)

            run_metric_values, run_gtr_metric_values = [], []
            results = load_run_results(result_fname)
            for tomo, res in results.items():
                if tomo.startswith("Tomo"):
                    metric_value = res.get(metric)
                    if np.isnan(metric_value):
                        continue

                    if metric not in ["Relative recall", "Total time taken"]:
                        gtr_metric_value = res.get(f"GTR {metric}")
                        if np.isnan(gtr_metric_value):
                            continue
                        run_gtr_metric_values.append(gtr_metric_value)

                    run_metric_values.append(metric_value)

            run_metric_values = np.array(run_metric_values)
            run_gtr_metric_values = np.array(run_gtr_metric_values)
            metric_values.append(np.expand_dims(run_metric_values, axis=1))
            gtr_metric_values.append(np.expand_dims(run_gtr_metric_values, axis=1))

        metric_values = np.concatenate(metric_values, axis=1)
        gtr_metric_values = np.concatenate(gtr_metric_values, axis=1)

        ### Statistics block: Compare against GTR
        comparison_significance_marks = []
        if metric not in ["Relative recall", "Total time taken"]:
            res2, gtr2 = metric_values.T, gtr_metric_values.T
            comparison_pvalues = []
            comparison_significance_marks = []
            for res_i, gtr_i in zip(res2, gtr2):
                u_stat, pvalue = mannwhitneyu(res_i, gtr_i)
                comparison_pvalues.append(float(pvalue))
                comparison_significance_marks.append(get_significance_mark(pvalue))

        ## Plotting
        fig, ax = plt.subplots(1, figsize=(6, 2))
        results_vp = ax.violinplot(
            metric_values, orientation="horizontal", showextrema=False
        )
        for idx, body in enumerate(results_vp["bodies"]):  # type:ignore
            body.set_facecolor(COLORS[workflow_keys[idx]])
            if "KM" in workflow_names[idx]:
                body.set_alpha(0.4)
            elif "GMM" in workflow_names[idx]:
                body.set_alpha(0.2)

        if metric not in ["Relative recall", "Total time taken"]:
            gtr_results_vp = ax.violinplot(
                gtr_metric_values, orientation="horizontal", showextrema=False
            )
            for idx, body in enumerate(gtr_results_vp["bodies"]):  # type:ignore
                body.set_facecolor("#aeaeae")
                body.set_alpha(0.4)

        results_bxplt = plt.boxplot(
            metric_values, orientation="horizontal", showfliers=True, patch_artist=True
        )
        for idx, mdn in enumerate(results_bxplt["medians"]):
            mdn.set(color="#000000")
            if metric not in ["Relative recall", "Total time taken"]:
                ax.text(
                    x=1.025,
                    y=idx + 0.97,
                    s=comparison_significance_marks[idx],
                    va="center",
                )

        for idx, flier_set in enumerate(results_bxplt["fliers"]):
            flier_set.set_marker("o")
            flier_set.set_markerfacecolor(
                COLORS[workflow_keys[idx]],
            )
            flier_set.set_markersize(3)
            flier_set.set_markeredgecolor("none")
        for bx in results_bxplt["boxes"]:
            bx.set_facecolor("none")

        if metric not in ["Relative recall", "Total time taken"]:
            gtr_results_bxplt = plt.boxplot(
                gtr_metric_values,
                orientation="horizontal",
                showfliers=True,
                flierprops=dict(
                    marker="o",
                    markerfacecolor="#aeaeae",
                    markersize=3,
                    markeredgecolor="none",
                ),
            )
            for mdn in gtr_results_bxplt["medians"]:
                mdn.set(color="#000000")

        ax.set_yticks([1, 2], workflow_names)
        if metric != "Total time taken":
            ax.set_xlim(0, 1)
            ax.set_xticks([i / 10 for i in range(11)])
            ax.set_xlabel(metric)
        else:
            ax.set_xlabel(f"{metric} (minutes)")

        plt.tight_layout()
        output_path = os.path.join(
            output_dir, f"mainfig_{dataset_id}_{metric}_comparison.png"
        )
        plt.savefig(output_path, dpi=600)
        plt.close()


if __name__ == "__main__":
    main()
