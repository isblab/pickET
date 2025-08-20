import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def main():
    particlewise_recall_fname = sys.argv[1]
    particle_details_fname = sys.argv[2]
    output_dir = sys.argv[3]
    x_keys = ["MW_pdb", "Rg", "Solidity"]

    with open(particlewise_recall_fname, "r") as input_f:
        particlewise_recall = yaml.safe_load(input_f)
    with open(particle_details_fname, "r") as pdeets_f:
        particle_details = yaml.safe_load(pdeets_f)["particles"]

    feature_extraction_method, clustering_method, particle_extraction_method = (
        "",
        "",
        "",
    )
    out_stats_dict = {}
    for x_key in x_keys:
        all_x, all_y = [], []
        for particle_id, results in particlewise_recall.items():
            all_x.append(particle_details[particle_id][x_key])
            all_y.append(results["Average recall"])

        all_x, all_y = np.array(all_x), np.array(all_y)
        pearsons_r, p_value = pearsonr(all_x, all_y)
        out_stats_dict[x_key] = {
            "pearsons_r": float(pearsons_r),  # type:ignore
            "p-value": float(p_value),  # type:ignore
        }
        plt.scatter(x=all_x, y=all_y, color="Orange", alpha=0.75, s=5)
        plt.ylim(0, 1)

        if x_key == "MW_pdb":
            plt.xlabel("Molecular weight obtained from PDB (kDa)")
        elif x_key == "Rg":
            plt.xlabel("Radius of gyration (Å)")
        elif x_key == "Solidity":
            plt.xlabel("Solidity")

        plt.ylabel("Average recall")

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

        outcsv_fname = f"{os.path.join(output_dir, f'source_data_particlewise_recall_{feature_extraction_method}_{clustering_method}_{particle_extraction_method}_{x_key}')}.csv"
        all_x_arr, all_y_arr = (
            np.expand_dims(all_x, axis=0),
            np.expand_dims(all_y, axis=0),
        )
        comb_arr = np.concatenate((all_x_arr, all_y_arr), axis=0)
        comb_arr = comb_arr.T
        cols = [x_key, "Average recall"]
        df = pd.DataFrame(comb_arr, columns=cols)
        df.to_csv(outcsv_fname, sep=",", index=False)

    with open(
        os.path.join(
            output_dir,
            f"particlewise_recall_{feature_extraction_method}_{clustering_method}_{particle_extraction_method}_stats.yaml",
        ),
        "w",
    ) as outf:
        yaml.dump(out_stats_dict, outf)


if __name__ == "__main__":
    main()
