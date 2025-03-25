import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt


def main():
    particlewise_recall_fname = sys.argv[1]
    x_key = "Rg"

    with open(particlewise_recall_fname, "r") as input_f:
        particlewise_recall = yaml.safe_load(input_f)

    all_x, all_y = [], []
    for particle_id, results in particlewise_recall.items():
        all_x.append(results[x_key])
        all_y.append(results["Average_recall"])

    all_x, all_y = np.array(all_x), np.array(all_y)
    # Get best fit line
    slope, intercept = np.polyfit(all_x, all_y, 1)
    yfit = slope * all_x + intercept

    # Get R square value
    sum_squared_residuals = np.sum((all_y - yfit) ** 2)
    sum_squared_total = np.sum((all_y - np.mean(all_y)) ** 2)
    r_squared_value = 1 - (sum_squared_residuals / sum_squared_total)

    plt.scatter(x=all_x, y=all_y, color="Orange", alpha=0.75, s=3)
    plt.plot(all_x, yfit, color="Green", alpha=0.75)
    plt.xlabel("Radius of gyration (A)")
    plt.ylabel("Average recall")
    plt.text(max(all_x) * 0.75, max(all_y), f"R-squared: {r_squared_value:.4f}")

    plt.savefig(
        os.path.join(
            "/home/shreyas/Projects/mining_tomograms/github/pickET/particlewise_recall/",
            f"particlewise_recall_{x_key}.png",
        ),
        dpi=600,
    )


if __name__ == "__main__":
    main()
