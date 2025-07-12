import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt


def main():
    result_fnames = sys.argv[1:]

    f1scores, relative_recalls = [], []
    for resfn in result_fnames:
        with open(resfn, "r") as f:
            run_results = yaml.safe_load(f)

        for tomo, m in run_results.items():
            if tomo.startswith("Tomo"):
                f1scores.append(m["F1-score"])
                relative_recalls.append(m["Relative recall"])

    f1scores, relative_recalls = np.array(f1scores), np.array(relative_recalls)
    # Get best fit line
    slope, intercept = np.polyfit(f1scores, relative_recalls, 1)
    yfit = slope * f1scores + intercept
    # Get R square value
    sum_squared_residuals = np.sum((relative_recalls - yfit) ** 2)
    sum_squared_total = np.sum((relative_recalls - np.mean(relative_recalls)) ** 2)
    r_squared_value = 1 - (sum_squared_residuals / sum_squared_total)

    plt.plot(
        [i / 10 for i in range(11)],
        [i / 10 for i in range(11)],
        linestyle="--",
        color="#aeaeae",
        label="One-to-one line",
    )
    plt.scatter(x=f1scores, y=relative_recalls, color="Orange", s=2.5)
    plt.plot(f1scores, yfit, color="Green", alpha=0.75, label="Best fit line")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("F1-scores")
    plt.ylabel("Relative recall")
    plt.text(max(f1scores) * 0.75, 0.1, f"Slope: {slope:.4f}")
    plt.text(max(f1scores) * 0.75, 0.16, f"Y-intercept: {intercept:.4f}")
    plt.text(max(f1scores) * 0.75, 0.22, f"R-squared: {r_squared_value:.4f}")
    plt.legend()
    plt.savefig("Comparison between F1-score and relative recall.png")


if __name__ == "__main__":
    main()
