import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scikit_posthocs import posthoc_dunn
from scipy.stats import kruskal, mannwhitneyu


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

METRICS = ["Precision", "Recall", "F1-score", "Relative recall", "Total time taken"]


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


def get_significance_matrix(pvalues: np.ndarray):
    out_array = np.zeros(pvalues.shape, dtype=object)
    for i in range(pvalues.shape[0]):
        for j in range(pvalues.shape[1]):
            if i == j:
                out_array[i, j] = "-"
            else:
                out_array[i, j] = get_significance_mark(pvalues[i, j])

    return out_array


def main():
    dataset_id = sys.argv[1]
    output_dir = sys.argv[2]
    result_fnames = sys.argv[3:]

    for metric in METRICS:
        xvals = []
        results = []
        gtr_results = []
        plot_time = False
        if (
            "10301" in result_fnames[0]
            or "tomotwin_efficiency_runs" in result_fnames[0]
        ):
            plot_time = True

        for idx, result_fname in enumerate(result_fnames):
            with open(result_fname, "r") as f:
                run_results = yaml.safe_load(f)

            metric_values = []
            gtr_metric_values = []
            for tomo, m in run_results.items():
                if tomo != "Global":
                    if metric in ["Precision", "Recall", "F1-score", "Relative recall"]:
                        if not np.isnan(m[metric]):
                            metric_values.append(m[metric])
                            if metric != "Relative recall":
                                gtr_metric_values.append(m[f"GTR {metric}"])

                    else:
                        if plot_time and metric == "Total time taken":
                            metric_values.append(m[metric] / 60)

            if len(metric_values) != 0:
                results.append(np.array(metric_values))
            if len(gtr_metric_values) != 0:
                gtr_results.append(np.array(gtr_metric_values))

            runname = "_".join(result_fname.split("/")[-3:])
            xvals.append(get_key_val(runname))

        if len(results) == 0:
            continue

        ### Statistics block
        results_arr = results.copy()
        gtr_results_arr = gtr_results.copy()

        # Compare against GTR
        gtr_comparison_pvalues = []
        for res_i, gtr_i in zip(results_arr, gtr_results_arr):
            u_stat, pvalue = mannwhitneyu(res_i, gtr_i)
            gtr_comparison_pvalues.append(pvalue)

        # Compare against each other
        h_stat, pvalue = kruskal(*results_arr)
        print(
            f"Kruskal-Wallis H-statistic and p-value on {metric} were found to be {h_stat:.4f} and {pvalue:.4f} for a comparison between all results"
        )
        if pvalue < 0.05:  # If significant Kruskal-Wallis, perform pairwise comparison
            names = []
            full_data = np.concatenate(results_arr, axis=0)
            for idx, res in enumerate(results_arr):
                names.extend([WORKFLOWS[xvals[idx]]] * len(res))
            names = np.array(names)

            df = pd.DataFrame({"Metric values": full_data, "Workflows": names})
            dunns_test_pval = posthoc_dunn(
                df,
                val_col="Metric values",
                group_col="Workflows",
                p_adjust="bonferroni",
            )

            significance_matrix = get_significance_matrix(dunns_test_pval.to_numpy())

            fig, ax = plt.subplots()
            ax.vlines(
                x=[i for i in range(1, significance_matrix.shape[0])],
                ymin=0,
                ymax=significance_matrix.shape[1],
                color="#aeaeae",
                linewidth=0.5,
            )
            ax.hlines(
                y=[i for i in range(1, significance_matrix.shape[0])],
                xmin=0,
                xmax=significance_matrix.shape[1],
                color="#aeaeae",
                linewidth=0.5,
            )
            for i in range(significance_matrix.shape[0]):
                for j in range(significance_matrix.shape[1]):
                    ax.text(
                        i + 0.5,
                        j + 0.5,
                        significance_matrix[i, j],
                        ha="center",
                        va="center",
                        fontdict={"size": 8},
                    )

            ax.set_xlim(0, significance_matrix.shape[0])
            ax.set_ylim(0, significance_matrix.shape[0])
            ax.set_xticks(
                [i + 0.5 for i in range(significance_matrix.shape[0])],
                [WORKFLOWS[i] for i in xvals],
                rotation=90,
            )
            ax.set_yticks(
                [i + 0.5 for i in range(significance_matrix.shape[0])],
                [WORKFLOWS[i] for i in xvals],
            )
            plt.tight_layout()
            output_path = os.path.join(
                output_dir, f"{dataset_id}_{metric}_comparison_significance.png"
            )
            plt.savefig(output_path, dpi=600)
            plt.close()

        ### Plotting part
        results_vp = plt.violinplot(results, vert=False, showextrema=False)
        for idx, body in enumerate(results_vp["bodies"]):  # type:ignore
            body.set_facecolor(COLORS[xvals[idx]])
            if xvals[idx].split("_")[-2] == "KM":
                body.set_alpha(0.4)
            elif xvals[idx].split("_")[-2] == "GMM":
                body.set_alpha(0.2)

        bxplt = plt.boxplot(
            results,
            orientation="horizontal",
            showfliers=True,
            flierprops=dict(
                marker="o",
                markerfacecolor="#505050",
                markersize=3,
                markeredgecolor="none",
            ),
        )  # type:ignore
        for mdn in bxplt["medians"]:
            mdn.set(color="#000000")

        if metric in ("Precision", "Recall", "F1-score"):
            random_results_vp = plt.violinplot(
                gtr_results, vert=False, showextrema=False
            )

            for body in random_results_vp["bodies"]:  # type:ignore
                body.set_facecolor("#aeaeae")
                body.set_alpha(0.75)

            for idx, whisker in enumerate(bxplt["whiskers"]):
                if idx % 2 == 1:  # Get only the top whisker
                    plt.text(
                        x=1.025,
                        y=(idx // 2) + 0.94,
                        s=get_significance_mark(gtr_comparison_pvalues[idx // 2]),
                        ha="left",
                        va="center",
                        fontdict={"size": 8},
                    )
        if plot_time and metric == "Total time taken":
            for idx in range(len(bxplt["medians"])):
                plt.text(
                    x=np.max(results) + 10 + 1,
                    y=(idx) + 0.98,
                    s=f"n={len(results[idx])}",
                    ha="left",
                    va="center",
                    fontdict={"size": 8},
                )
        elif metric in ("Precision", "Recall", "F1-score"):
            for idx in range(len(bxplt["medians"])):
                plt.text(
                    x=1.1,
                    y=(idx) + 0.98,
                    s=f"n={len(results[idx])}",
                    ha="left",
                    va="center",
                    fontdict={"size": 8},
                )
        else:
            for idx in range(len(bxplt["medians"])):
                plt.text(
                    x=1.025,
                    y=(idx) + 0.98,
                    s=f"n={len(results[idx])}",
                    ha="left",
                    va="center",
                    fontdict={"size": 8},
                )

        xlabels = [WORKFLOWS[k] for k in xvals]

        if metric in ("Precision", "Recall", "F1-score", "Relative recall"):
            plt.xlim(0, 1)
        if plot_time and metric == "Total time taken":
            plt.xlim(0, np.max(results) + 10)

        plt.yticks(range(1, len(xlabels) + 1), xlabels)
        if metric == "Total time taken":
            plt.xlabel("Total time (min)")
        else:
            plt.xlabel(metric)

        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{dataset_id}_{metric}_comparison.png")
        plt.savefig(output_path, dpi=600)
        plt.close()


if __name__ == "__main__":
    main()
