import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


def get_best_fit_line(metric1, metric2):
    # Get best fit line
    slope, intercept = np.polyfit(metric1, metric2, 1)
    yfit = slope * metric1 + intercept
    # Get R square value
    sum_squared_residuals = np.sum((metric2 - yfit) ** 2)
    sum_squared_total = np.sum((metric2 - np.mean(metric2)) ** 2)
    r_squared_value = 1 - (sum_squared_residuals / sum_squared_total)
    return yfit, slope, intercept, r_squared_value


def plot(x_metric, y_metric, x_metric_name, y_metric_name, run_name):
    yfit, slope, intercept, r_squared_value = get_best_fit_line(x_metric, y_metric)

    # xvals = np.arange(0, 10, 0.1) / 10
    # plt.plot(xvals, xvals, color="#aeaeae", alpha=0.75, label="y=x line")
    plt.scatter(x=x_metric, y=y_metric, color="Orange", s=2.5)
    plt.plot(x_metric, yfit, color="Green", alpha=0.75, label="Best fit line")
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.xlabel(x_metric_name)
    plt.ylabel(y_metric_name)
    plt.text(max(x_metric) * 0.75, 0.1, f"Slope: {slope:.4f}")
    plt.text(max(x_metric) * 0.75, 0.16, f"Y-intercept: {intercept:.4f}")
    plt.text(max(x_metric) * 0.75, 0.22, f"R-squared: {r_squared_value:.4f}")
    plt.legend()
    # plt.savefig(
    #     f"Comparison between {x_metric_name} and {y_metric_name}_{run_name}_run.png"
    # )
    plt.show()
    plt.close()


def main():
    run_name = sys.argv[1]
    result_fnames = sys.argv[2:]

    precisions, recalls, f1scores = [], [], []
    mdrs, one_minus_mdrs, relative_recalls = [], [], []
    delta_recalls = []

    for resfn in result_fnames:
        with open(resfn, "r") as f:
            run_results = yaml.safe_load(f)

        for tomo, m in run_results.items():
            if tomo.startswith("Tomo"):
                precisions.append(m["Precision"])
                recalls.append(m["Recall"])
                f1scores.append(m["F1-score"])

                mdrs.append(m["MDR Recall"])
                one_minus_mdrs.append(1 - m["MDR Recall"])
                relative_recalls.append(m["Relative recall"])

                dr = (m["Recall"] - m["MDR Recall"] + 1) / 2
                delta_recalls.append(dr)

    f1scores, relative_recalls = (
        np.array(f1scores),
        np.array(relative_recalls),
    )
    precisions, one_minus_mdrs = (
        np.array(precisions),
        np.array(one_minus_mdrs),
    )
    delta_recalls = np.array(delta_recalls)

    out_table = []
    for i in range(len(f1scores)):
        out_table.append(
            {
                "Precision": precisions[i],
                "Recall": recalls[i],
                "F1-score": f1scores[i],
                "MDR recall": mdrs[i],
                "1-MDR": one_minus_mdrs[i],
                "Relative recall": relative_recalls[i],
                "Delta recall": delta_recalls[i],
            }
        )
    print(tabulate(out_table, headers="keys"))

    plot(f1scores, relative_recalls, "F1-score", "Relative recall", run_name)
    plot(precisions, one_minus_mdrs, "Precision", "1-MDR", run_name)
    plot(f1scores, delta_recalls, "F1-score", "Delta recall", run_name)


if __name__ == "__main__":
    main()
