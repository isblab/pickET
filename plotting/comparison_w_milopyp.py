import os
import sys
import glob
import yaml
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from picket.core import utils


def read_selected_workflows_yaml(fpath: str) -> dict[str, str]:
    with open(fpath, "r") as yamlf:
        selected_workflows = yaml.safe_load(yamlf)
    return selected_workflows


def load_all_results_files(fpaths: list[str]) -> dict:
    all_results = {}
    for fpath in fpaths:
        with open(fpath, "r") as resf:
            all_results[fpath] = yaml.safe_load(resf)
    return all_results


def set_boxplot_visualization(
    bxplt_m, bxplt_p, milopyp_color: str, picket_color: str, alpha: float
):
    for idx, bxplt in enumerate((bxplt_m, bxplt_p)):
        if idx == 0:
            color = milopyp_color
        else:
            color = picket_color
        for bx in bxplt["boxes"]:
            bx.set_edgecolor(color=color)
            bx.set_facecolor(color=color)
            bx.set_alpha(alpha)
        for md in bxplt["medians"]:
            md.set_color(color=color)
        for ws in bxplt["whiskers"]:
            ws.set_color(color=color)
        for cp in bxplt["caps"]:
            cp.set_color(color=color)


def main():
    milopyp_parent_path = sys.argv[1]
    picket_runs_parent_path = sys.argv[2]
    # selected_runs_fpath = sys.argv[3]
    # selected_picket_workflows = read_selected_workflows_yaml(selected_runs_fpath)

    tomotwin_milopyp_pr, tomotwin_milopyp_re, tomotwin_milopyp_f1 = [], [], []
    tomotwin_picket_pr, tomotwin_picket_re, tomotwin_picket_f1 = [], [], []

    all_milopyp_recall, all_picket_recall = [], []
    all_milopyp_relativerecall, all_picket_relativerecall = [], []
    milopyp_total_time_taken_10301 = []
    picket_total_time_taken_10301_gabor_gmm_ws = []
    picket_total_time_taken_10301_library_based = []

    xlabels = []
    datasets = ["10001", "10008", "10301", "10440", "tomotwin"]

    for dataset_id in datasets:
        if dataset_id == "tomotwin":
            xlabels.append("TomoTwin")
        else:
            xlabels.append(dataset_id)
        if dataset_id.startswith("10"):
            picket_result_fnames = glob.glob(
                os.path.join(
                    picket_runs_parent_path,
                    f"czi_ds-{dataset_id}_denoised",
                    "evaluations",
                    "overall_results_*.yaml",
                )
            )
        elif dataset_id.startswith("tomotw"):
            picket_result_fnames = glob.glob(
                os.path.join(
                    picket_runs_parent_path,
                    dataset_id,
                    "evaluations",
                    "overall_results_*.yaml",
                )
            )
        else:
            raise ValueError(f"Unknown dataset ID detected: {dataset_id}")

        all_picket_results = load_all_results_files(picket_result_fnames)

        milopyp_eval_path = os.path.join(
            milopyp_parent_path,
            dataset_id,
            "evaluations",
            "overall_results_pred_coords.yaml_.yaml",
        )

        with open(milopyp_eval_path, "r") as mep:
            milopyp_evals = yaml.safe_load(mep)

        milopyp_recall, milopyp_relativerecall = [], []
        picket_recall, picket_relativerecall = [], []

        for tomo_id in milopyp_evals:
            if tomo_id.startswith("Tomo_ID"):
                m_recall = float(milopyp_evals[tomo_id]["Recall"])
                m_relative_recall = float(
                    np.sqrt(m_recall * (1 - milopyp_evals[tomo_id]["MDR Recall"]))
                )

                # numerical_tomo_id = int(tomo_id.split(" - ")[-1])
                # sel_picket_workflow = selected_picket_workflows[dataset_id][
                #     numerical_tomo_id
                # ]
                if dataset_id == "tomotwin":
                    sel_picket_workflow = os.path.join(
                        picket_runs_parent_path,
                        dataset_id,
                        "evaluations",
                        "overall_results_gabor_gmm_connected_component_labeling.yaml",
                    )
                elif dataset_id.startswith("10"):
                    sel_picket_workflow = os.path.join(
                        picket_runs_parent_path,
                        f"czi_ds-{dataset_id}_denoised",
                        "evaluations",
                        "overall_results_gabor_kmeans_watershed_segmentation.yaml",
                    )
                else:
                    raise ValueError("Unrecognised dataset found")

                picket_res = all_picket_results[sel_picket_workflow][tomo_id]

                if dataset_id == "10301":
                    milopyp_total_time_taken_10301.append(
                        milopyp_evals[tomo_id]["Total time taken"]
                    )

                    picket_time_taken_for_s1 = 0
                    picket_time_taken_for_s2 = 0
                    for k, pr in all_picket_results.items():
                        # k is workflow name and pr is contents of the file
                        if "connected_component_labeling" in k:
                            picket_time_taken_for_s1 += pr[tomo_id]["Time taken for S1"]

                    pth = os.path.join(*sel_picket_workflow.split("/")[:-1])
                    fname1 = "_".join(sel_picket_workflow.split("/")[-1].split("_")[:4])
                    fname2 = (
                        "connected_component_labeling.yaml",
                        "watershed_segmentation.yaml",
                    )
                    for f in fname2:
                        fname = f"/{os.path.join(pth, f'{fname1}_{f}')}"
                        picket_time_taken_for_s2 += all_picket_results[fname][tomo_id][
                            "Time taken for S1"
                        ]

                    picket_total_time_taken_10301_library_based.append(
                        picket_time_taken_for_s1 + picket_time_taken_for_s2
                    )

                    # -------------------------------------------
                    picket_total_time_taken_10301_gabor_gmm_ws.append(
                        float(picket_res["Total time taken"])
                    )

                p_recall = float(picket_res["Recall"])
                p_relative_recall = float(
                    np.sqrt(p_recall * (1 - picket_res["MDR Recall"]))
                )
                # if p_relative_recall < 0.1:
                #     print(
                #         dataset_id,
                #         tomo_id,
                #         sel_picket_workflow,
                #         p_recall,
                #         picket_res["MDR Recall"],
                #         (1 - picket_res["MDR Recall"]),
                #         p_recall * (1 - picket_res["MDR Recall"]),
                #         p_relative_recall,
                #     )

                if not math.isnan(m_recall) and not math.isnan(p_recall):
                    milopyp_recall.append(m_recall)
                    picket_recall.append(p_recall)
                if not math.isnan(m_relative_recall) and not math.isnan(
                    p_relative_recall
                ):
                    milopyp_relativerecall.append(m_relative_recall)
                    picket_relativerecall.append(p_relative_recall)

                if dataset_id == "tomotwin":
                    tomotwin_milopyp_pr.append(milopyp_evals[tomo_id]["Precision"])
                    tomotwin_milopyp_re.append(milopyp_evals[tomo_id]["Recall"])
                    tomotwin_milopyp_f1.append(milopyp_evals[tomo_id]["F1-score"])

                    tomotwin_picket_pr.append(picket_res["Precision"])
                    tomotwin_picket_re.append(picket_res["Recall"])
                    tomotwin_picket_f1.append(picket_res["F1-score"])

        all_milopyp_recall.append(milopyp_recall)
        all_picket_recall.append(picket_recall)
        all_milopyp_relativerecall.append(milopyp_relativerecall)
        all_picket_relativerecall.append(picket_relativerecall)

    tomotwin_milopyp_metrics = np.concatenate(
        [
            np.expand_dims(np.array(tomotwin_milopyp_f1), axis=1),
            np.expand_dims(np.array(tomotwin_milopyp_re), axis=1),
            np.expand_dims(np.array(tomotwin_milopyp_pr), axis=1),
        ],
        axis=1,
    )
    tomotwin_picket_metrics = np.concatenate(
        [
            np.expand_dims(np.array(tomotwin_picket_f1), axis=1),
            np.expand_dims(np.array(tomotwin_picket_re), axis=1),
            np.expand_dims(np.array(tomotwin_picket_pr), axis=1),
        ],
        axis=1,
    )

    ### Plotting part
    milopyp_color = "#AEAEAE"
    picket_color = "#48A3F8"
    alpha = 0.5
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(5, 2, fig)

    ax0 = fig.add_subplot(gs[0, 0])
    bxplt_m = ax0.boxplot(
        tomotwin_milopyp_metrics,
        orientation="horizontal",
        showfliers=False,
        patch_artist=True,
    )
    bxplt_p = ax0.boxplot(
        tomotwin_picket_metrics,
        orientation="horizontal",
        showfliers=False,
        patch_artist=True,
    )
    set_boxplot_visualization(bxplt_m, bxplt_p, milopyp_color, picket_color, alpha)
    ax0.set_yticks(
        range(1, tomotwin_picket_metrics.shape[1] + 1),
        ["F1-score", "Recall", "Precision"],
    )
    ax0.set_ylabel("Metrics")
    ax0.set_xlabel("Metric values")
    ax0.set_xlim(0, 1)

    ax1 = fig.add_subplot(gs[1, 0])
    bxplt_m = ax1.boxplot(
        all_milopyp_relativerecall,
        orientation="horizontal",
        showfliers=False,
        patch_artist=True,
    )
    bxplt_p = ax1.boxplot(
        all_picket_relativerecall,
        orientation="horizontal",
        showfliers=False,
        patch_artist=True,
    )

    set_boxplot_visualization(bxplt_m, bxplt_p, milopyp_color, picket_color, alpha)
    ax1.set_yticks(range(1, len(xlabels) + 1), xlabels)
    ax1.set_ylabel("Dataset")
    ax1.set_xlabel("Relative recall")
    ax1.set_xlim(0, 1)

    color_palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#5C4033"]

    ax2 = fig.add_subplot(gs[2:4, 0])
    for idx, (mrrs, prrs) in enumerate(
        zip(all_milopyp_relativerecall, all_picket_relativerecall)
    ):
        ax2.scatter(
            mrrs, prrs, color=color_palette[idx], label=datasets[idx], alpha=0.7
        )

    xys = np.linspace(0, 1, 10)
    ax2.plot(xys, xys, color="#aeaeae", alpha=0.4, linestyle="--")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("MiLoPYP relative recall")
    ax2.set_ylabel("PickET relative recall")

    ax3 = fig.add_subplot(gs[0, 1])
    ax3.barh(
        y=["PickET library", "PickET\nGabor-GMM-WS", "MiLoPYP"],
        width=[
            np.sum(np.array(picket_total_time_taken_10301_library_based)) / 60,
            np.sum(np.array(picket_total_time_taken_10301_gabor_gmm_ws)) / 60,
            milopyp_total_time_taken_10301[0] / 60,
        ],
        height=0.5,
        color=[picket_color, picket_color, milopyp_color],
        alpha=0.75,
    )
    ax3.set_xlabel("Total time taken (minutes)")
    print(
        np.sum(np.array(picket_total_time_taken_10301_library_based)) / 60,
        np.sum(np.array(picket_total_time_taken_10301_gabor_gmm_ws)) / 60,
        milopyp_total_time_taken_10301[0] / 60,
    )

    # ax4 = fig.add_subplot(gs[0:2, 1])
    # tomo = utils.load_tomogram(
    #     "/data2/shreyas/mining_tomograms/datasets/cryoet_dataportal/10301/tomogram_14318/01122021_BrnoKrios_arctis_lam2_pos13.mrc"
    # )[0]
    # ax4.imshow(tomo[tomo.shape[0] // 2], cmap="Greys")
    # ax4.set_xticks([])
    # ax4.set_yticks([])

    plt.tight_layout()

    plt.savefig(
        "/home/shreyas/Dropbox/miningTomograms/manuscript/comparison_w_milopyp/template_comparison_w_milopyp_tomogram-wise.png",
        dpi=1200,
    )


if __name__ == "__main__":
    main()
