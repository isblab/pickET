import sys
import yaml
import numpy as np


def main():
    particlewise_recall_fname = sys.argv[1]
    particle_deets_fname = sys.argv[2]

    with open(particlewise_recall_fname, "r") as pwrf:
        particlewise_recall = yaml.safe_load(pwrf)
    with open(particle_deets_fname, "r") as pdeetf:
        particle_deets = yaml.safe_load(pdeetf)["particles"]

    recall_greater_than_half = 0
    n_particle_types = 0

    small_recall_greater_than_half = 0
    small_n_particle_types = 0

    big_recall_greater_than_half = 0
    big_n_particle_types = 0

    convex_recall_greater_than_half = 0
    convex_n_particle_types = 0
    not_convex_recall_greater_than_half = 0
    not_convex_n_particle_types = 0

    mws = []
    for pdbid, metricvals in particlewise_recall.items():
        if float(metricvals["Average recall"]) > 0.5:
            recall_greater_than_half += 1

        n_particle_types += 1

        if particle_deets[pdbid]["MW_pdb"] < 500:
            small_n_particle_types += 1
            if float(metricvals["Average recall"]) > 0.5:
                small_recall_greater_than_half += 1

        if particle_deets[pdbid]["MW_pdb"] >= 500:
            big_n_particle_types += 1
            if float(metricvals["Average recall"]) > 0.5:
                big_recall_greater_than_half += 1

        if particle_deets[pdbid]["Solidity"] >= 0.75:
            convex_n_particle_types += 1
            if float(metricvals["Average recall"]) > 0.5:
                convex_recall_greater_than_half += 1
        if particle_deets[pdbid]["Solidity"] < 0.75:
            not_convex_n_particle_types += 1
            if float(metricvals["Average recall"]) > 0.5:
                not_convex_recall_greater_than_half += 1

        mws.append(particle_deets[pdbid]["MW_pdb"])

    print(
        recall_greater_than_half,
        n_particle_types,
        recall_greater_than_half / n_particle_types,
    )
    print(np.min(np.array(mws)), np.max(np.array(mws)))
    print(
        small_recall_greater_than_half,
        small_n_particle_types,
        small_recall_greater_than_half / small_n_particle_types,
    )
    print(
        big_recall_greater_than_half,
        big_n_particle_types,
        big_recall_greater_than_half / big_n_particle_types,
    )

    print(
        convex_recall_greater_than_half,
        convex_n_particle_types,
        convex_recall_greater_than_half / convex_n_particle_types,
    )
    print(
        not_convex_recall_greater_than_half,
        not_convex_n_particle_types,
        not_convex_recall_greater_than_half / not_convex_n_particle_types,
    )


if __name__ == "__main__":
    main()
