import yaml
import numpy as np
import matplotlib.pyplot as plt


def get_relative_recalls(results: dict) -> np.ndarray:
    out_relative_recalls = []
    for tomo_head, tomo_evals in results.items():
        if not tomo_head.startswith("Global"):
            recall = tomo_evals["Recall"]
            mdr_recall = tomo_evals["MDR Recall"]
            relative_recall = float(np.sqrt(recall * (1 - mdr_recall)))
            out_relative_recalls.append(relative_recall)
    return np.array(out_relative_recalls)


def main():
    milopyp_eval_path = "/data2/shreyas/mining_tomograms/working/s1_clean_results_picket_v2/comparison_w_milopyp/untimed_milopyp_run/evaluations/overall_results_pred_coords.yaml_.yaml"
    picket_eval_path = "/data2/shreyas/mining_tomograms/working/s1_clean_results_picket_v2/comparison_w_milopyp/picket_run/evaluations/overall_results_picket_run.yaml"
    color_palette = {"MiLoPYP": "#e06666", "PickET": "#4a86e8"}
    with open(milopyp_eval_path, "r") as milopyp_f:
        milopyp_eval = yaml.safe_load(milopyp_f)
    with open(picket_eval_path, "r") as picket_f:
        picket_eval = yaml.safe_load(picket_f)

    milopyp_relative_recalls = get_relative_recalls(milopyp_eval)
    picket_relative_recalls = get_relative_recalls(picket_eval)

    mwins, pwins = 0, 0
    for m, p in zip(milopyp_relative_recalls, picket_relative_recalls):
        if m > p:
            mwins += 1
        elif m < p:
            pwins += 1
        else:
            mwins += 1
            pwins += 1

    fig, ax = plt.subplots(1, 4)
    ax[0].scatter(
        x=[i + 1 for i in range(len(milopyp_relative_recalls))],
        y=milopyp_relative_recalls,
        color="#e06666",
        alpha=0.75,
        label="MiLoPYP predictions",
    )
    ax[0].scatter(
        x=[i + 1 for i in range(len(picket_relative_recalls))],
        y=picket_relative_recalls,
        color="#4a86e8",
        alpha=0.75,
        label="PickET predictions",
    )
    ax[0].set_xlabel("Tomogram index")
    ax[0].set_ylabel("Relative recall")
    ax[0].set_ylim(0, 1)

    ax[1].scatter(x="MiLoPYP", y=mwins, color="#e06666")
    ax[1].scatter(x="PickET", y=pwins, color="#4a86e8")
    ax[1].set_xlabel("Method")
    ax[1].set_ylabel("Number of times it was better")
    ax[1].set_ylim(0, np.maximum(mwins, pwins) + 1)

    ax[2].scatter(x="MiLoPYP", y=np.mean(milopyp_relative_recalls), color="#e06666")
    ax[2].scatter(x="PickET", y=np.mean(picket_relative_recalls), color="#4a86e8")
    ax[2].set_xlabel("Method")
    ax[2].set_ylabel("Average relative recall")
    ax[2].set_ylim(0, 1)

    results = np.array([milopyp_relative_recalls, picket_relative_recalls]).T
    xvals = ["MiLoPYP", "PickET"]
    vp = ax[3].violinplot(results, vert=True, showextrema=False)
    for idx, body in enumerate(vp["bodies"]):  # type:ignore
        body.set_facecolor(color_palette[xvals[idx]])
    ax[3].boxplot(
        results,
        orientation="vertical",
        showfliers=False,
        boxprops=dict(color="#000000"),
        whiskerprops=dict(color="#000000"),
        capprops=dict(color="#000000"),
        medianprops=dict(color="#000000"),
    )  # type:ignore

    ax[3].set_xticks(range(1, len(xvals) + 1), xvals)
    ax[3].set_xlabel("Method")
    ax[3].set_ylabel("Relative recall")
    ax[3].set_ylim(0, 1)

    plt.show()


if __name__ == "__main__":
    main()
