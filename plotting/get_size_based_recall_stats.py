import os
import sys
import yaml


def main():
    particlewise_recall_fname = sys.argv[1]
    particle_details_fname = sys.argv[2]
    recall_threshold = 0.7
    mol_wt_threshold = 500

    with open(particlewise_recall_fname, "r") as input_f:
        particlewise_recall = yaml.safe_load(input_f)
    with open(particle_details_fname, "r") as pdeets_f:
        particle_details = yaml.safe_load(pdeets_f)["particles"]

    # -------------------------------------------------------------------------
    # -------------------------------- Claim 1 --------------------------------
    # -------------------------------------------------------------------------
    win_count1, total_count1 = 0, 0
    for pdb_id, metrics in particlewise_recall.items():
        if metrics["Average recall"] >= recall_threshold:
            win_count1 += 1
        total_count1 += 1
    percentage1 = (win_count1 / total_count1) * 100
    print("\nClaim 1:")
    print(
        f"{win_count1} out of {total_count1} total particles ({percentage1:.0f}%) were localized with recall >= {recall_threshold}"
    )
    print()

    # -------------------------------------------------------------------------
    # -------------------------------- Claim 2 --------------------------------
    # -------------------------------------------------------------------------
    win_count2, total_count2 = 0, 0
    for pdb_id, metrics in particlewise_recall.items():
        mol_wt = particle_details[pdb_id]["MW_pdb"]
        p_recall = metrics["Average recall"]
        if mol_wt >= mol_wt_threshold:
            total_count2 += 1
            if p_recall >= recall_threshold:
                win_count2 += 1

    percentage2 = (win_count2 / total_count2) * 100
    print("\nClaim 2:")
    print(
        f"{win_count2} out of {total_count2} ({percentage2:.0f}%) particles larger than or equal to {mol_wt_threshold} kDa were localized with recall >= {recall_threshold}"
    )
    print()

    # -------------------------------------------------------------------------
    # -------------------------------- Claim 3 --------------------------------
    # -------------------------------------------------------------------------
    win_count3, total_count3 = 0, 0
    for pdb_id, metrics in particlewise_recall.items():
        mol_wt = particle_details[pdb_id]["MW_pdb"]
        p_recall = metrics["Average recall"]
        if mol_wt < mol_wt_threshold:
            total_count3 += 1
            if p_recall >= recall_threshold:
                win_count3 += 1
    percentage3 = (win_count3 / total_count3) * 100
    print("\nClaim 3:")
    print(
        f"{win_count3} out of {total_count3} ({percentage3:.0f}%) particles larger than or equal to {mol_wt_threshold} kDa were localized with recall >= {recall_threshold}"
    )
    print()


if __name__ == "__main__":
    main()
