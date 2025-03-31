import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt


def main():
    particlewise_recall_fname = sys.argv[1]

    with open(particlewise_recall_fname, "r") as input_f:
        particlewise_recall = yaml.safe_load(input_f)

    all_x, all_y = [], []
    for particle_id, results in particlewise_recall.items():
        all_x.append(results["Average_recall"])
        all_y.append(particle_id)

    all_x, all_y = np.array(all_x), np.array(all_y)

    plt.barh(y=all_y, width=all_x, color="Orange", alpha=0.75)
    plt.xlim(0, 1)

    plt.xlabel("Particle")
    plt.ylabel("Average recall")
    # plt.xticks(rotation=85)
    plt.tight_layout()

    fname_head = particlewise_recall_fname.split("/")[-1][:-5]
    feature_extraction_method = fname_head.split("_")[2]
    clustering_method = fname_head.split("_")[3]
    particle_extraction_method = "_".join(fname_head.split("_")[4:])
    plt.savefig(
        os.path.join(
            "/home/shreyas/Dropbox/miningTomograms/particlewise_recall/",
            f"particlewise_recall_{feature_extraction_method}_{clustering_method}_{particle_extraction_method}.png",
        ),
        dpi=600,
    )
    plt.close()


if __name__ == "__main__":
    main()
