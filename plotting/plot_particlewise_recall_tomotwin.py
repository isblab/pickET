import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt


def main():
    particlewise_recall_fname = sys.argv[1]
    particle_details_fname = sys.argv[2]
    x_keys = ["MW_pdb", "Rg"]

    with open(particlewise_recall_fname, "r") as input_f:
        particlewise_recall = yaml.safe_load(input_f)
    with open(particle_details_fname, "r") as pdeets_f:
        particle_details = yaml.safe_load(pdeets_f)["particles"]

    for x_key in x_keys:
        all_x, all_y, all_yr = [], [], []
        for particle_id, results in particlewise_recall.items():
            all_x.append(particle_details[particle_id][x_key])
            all_y.append(results["Average recall"])
            all_yr.append(results["Average recall ratio"])

        all_x, all_y, all_yr = np.array(all_x), np.array(all_y), np.array(all_yr)
        # Get best fit line
        slope, intercept = np.polyfit(all_x, all_y, 1)
        slope_r, intercept_r = np.polyfit(all_x, all_yr, 1)
        yfit = slope * all_x + intercept
        yfitr = slope_r * all_x + intercept_r

        # Get R square value
        sum_squared_residuals = np.sum((all_y - yfit) ** 2)
        sum_squared_total = np.sum((all_y - np.mean(all_y)) ** 2)
        r_squared_value = 1 - (sum_squared_residuals / sum_squared_total)

        sum_squared_residuals_r = np.sum((all_yr - yfitr) ** 2)
        sum_squared_total_r = np.sum((all_yr - np.mean(all_yr)) ** 2)
        r_squared_value_r = 1 - (sum_squared_residuals_r / sum_squared_total_r)

        fig, ax = plt.subplots(1, 2)
        fig.set_figheight(5)
        fig.set_figwidth(10)
        ax[0].scatter(x=all_x, y=all_y, color="Orange", alpha=0.75, s=3)
        ax[0].plot(all_x, yfit, color="Green", alpha=0.75)
        ax[0].set_ylim(0, 1)

        ax[0].set_xlabel("Molecular weight obtained from PDB (kDa)")
        ax[0].set_ylabel("Average recall")
        ax[0].text(max(all_x) * 0.75, 0.1, f"Slope: {slope:.4f}")
        ax[0].text(max(all_x) * 0.75, 0.16, f"Y-intercept: {intercept:.4f}")
        ax[0].text(max(all_x) * 0.75, 0.22, f"R-squared: {r_squared_value:.4f}")

        ax[1].scatter(x=all_x, y=all_yr, color="Orange", alpha=0.75, s=3)
        ax[1].plot(all_x, yfitr, color="Green", alpha=0.75)

        ax[1].set_xlabel("Molecular weight obtained from PDB (kDa)")
        ax[1].set_ylabel("Average recall ratio")
        ax[1].text(max(all_x) * 0.75, 0.1, f"Slope: {slope:.4f}")
        ax[1].text(max(all_x) * 0.75, 0.16, f"Y-intercept: {intercept:.4f}")
        ax[1].text(max(all_x) * 0.75, 0.22, f"R-squared: {r_squared_value:.4f}")

        fname_head = particlewise_recall_fname.split("/")[-1][:-5]
        feature_extraction_method = fname_head.split("_")[2]
        clustering_method = fname_head.split("_")[3]
        particle_extraction_method = "_".join(fname_head.split("_")[4:])
        plt.savefig(
            os.path.join(
                "/home/shreyas/Projects/mining_tomograms/pickET/partice_wise_recall/TomoTwin/",
                f"particlewise_recall_{feature_extraction_method}_{clustering_method}_{particle_extraction_method}_{x_key}.png",
            ),
            dpi=600,
        )
        plt.close()


if __name__ == "__main__":
    main()
