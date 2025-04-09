import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt


def format_particle_name(name):
    out_str = []
    counter = 0
    for s in name.split("_"):
        out_str.append(s)
        counter += 1
        if counter % 2 == 0:
            out_str.append("\n")

    return " ".join(out_str)


def main():
    particlewise_recall_fname = sys.argv[1]
    dataset_id = sys.argv[2]

    with open(particlewise_recall_fname, "r") as input_f:
        particlewise_recall = yaml.safe_load(input_f)

    all_x, all_y = [], []
    for particle_id, results in particlewise_recall.items():
        all_x.append(results["Average recall"])
        particle_id = particle_id.split("-")[-2]
        particle_id = format_particle_name(particle_id)
        all_y.append(particle_id)

        recalls = results["Recalls"]
        plt.scatter(
            x=recalls,
            y=[particle_id for _ in recalls],
            color="Orange",
            alpha=0.75,
            zorder=2,
        )

    all_x, all_y = np.array(all_x), np.array(all_y)

    plt.barh(y=all_y, width=all_x, alpha=0.75, zorder=1)
    plt.xlim(0, 1)

    plt.xlabel("Average recall")
    plt.ylabel("Particle")
    plt.tight_layout()

    fname_head = particlewise_recall_fname.split("/")[-1][:-5]
    feature_extraction_method = fname_head.split("_")[2]
    clustering_method = fname_head.split("_")[3]
    particle_extraction_method = "_".join(fname_head.split("_")[4:])
    plt.savefig(
        os.path.join(
            f"/home/shreyas/Dropbox/miningTomograms/particlewise_recall/{dataset_id}/",
            f"particlewise_recall_{feature_extraction_method}_{clustering_method}_{particle_extraction_method}.png",
        ),
        dpi=600,
    )
    plt.close()


if __name__ == "__main__":
    main()
