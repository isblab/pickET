import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt


def main():
    particlewise_recall_fname = sys.argv[1]
    particle_details_fname = sys.argv[2]
    output_dir = sys.argv[3]
    x_keys = ["MW_pdb", "Rg"]

    with open(particlewise_recall_fname, "r") as input_f:
        particlewise_recall = yaml.safe_load(input_f)
    with open(particle_details_fname, "r") as pdeets_f:
        particle_details = yaml.safe_load(pdeets_f)["particles"]

    for x_key in x_keys:
        all_x, all_y = [], []
        for particle_id, results in particlewise_recall.items():
            all_x.append(particle_details[particle_id][x_key])
            all_y.append(results["Average recall"])

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
        plt.ylim(0, 1)

        print(x_key, np.max(all_x))
        if x_key == "MW_pdb":
            plt.xlabel("Molecular weight obtained from PDB (kDa)")
        elif x_key == "Rg":
            plt.xlabel("Radius of gyration (Å)")

        plt.ylabel("Average recall")
        plt.text(max(all_x) * 0.75, 0.1, f"Slope: {slope:.4f}")
        plt.text(max(all_x) * 0.75, 0.16, f"Y-intercept: {intercept:.4f}")
        plt.text(max(all_x) * 0.75, 0.22, f"R-squared: {r_squared_value:.4f}")

        fname_head = particlewise_recall_fname.split("/")[-1][:-5]
        feature_extraction_method = fname_head.split("_")[2]
        clustering_method = fname_head.split("_")[3]
        particle_extraction_method = "_".join(fname_head.split("_")[4:])
        plt.savefig(
            os.path.join(
                output_dir,
                f"particlewise_recall_{feature_extraction_method}_{clustering_method}_{particle_extraction_method}_{x_key}.png",
            ),
            dpi=600,
        )
        plt.close()


if __name__ == "__main__":
    main()
