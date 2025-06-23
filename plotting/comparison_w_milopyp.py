import os
import sys
import yaml
import matplotlib.pyplot as plt

milopyp_parent_path = sys.argv[1]
picket_parent_path = sys.argv[2]


all_milopyp_pr, all_milopyp_re, all_milopyp_f1 = [], [], []
all_milopyp_rpr, all_milopyp_rre, all_milopyp_rf1 = [], [], []
all_picket_pr, all_picket_re, all_picket_f1 = [], [], []
all_picket_rpr, all_picket_rre, all_picket_rf1 = [], [], []
xvals = []

for dataset_id in ["10001", "10008", "10301", "10440", "tomotwin"]:
    milopyp_eval_path = os.path.join(
        milopyp_parent_path,
        dataset_id,
        "evaluations",
        "overall_results_pred_coords.yaml_.yaml",
    )
    picket_eval_path = os.path.join(milopyp_parent_path, dataset_id, "evaluations")

    with open(milopyp_eval_path, "r") as mep:
        milopyp_evals = yaml.safe_load(mep)

    #! Make picket_eval_path come from a yaml file containing all selected workflows
    with open(picket_eval_path, "r") as pep:
        picket_evals = yaml.safe_load(pep)

    milopyp_pr, milopyp_re, milopyp_f1 = [], [], []
    milopyp_rpr, milopyp_rre, milopyp_rf1 = [], [], []
    picket_pr, picket_re, picket_f1 = [], [], []
    picket_rpr, picket_rre, picket_rf1 = [], [], []
    for tomo_id in milopyp_evals:
        if tomo_id.startswith("Tomo_ID"):
            milopyp_pr.append(milopyp_evals[tomo_id]["Precision"])
            milopyp_re.append(milopyp_evals[tomo_id]["Recall"])
            milopyp_f1.append(milopyp_evals[tomo_id]["F1-score"])

            milopyp_rpr.append(milopyp_evals[tomo_id]["Random Precision"])
            milopyp_rre.append(milopyp_evals[tomo_id]["Random Recall"])
            milopyp_rf1.append(milopyp_evals[tomo_id]["Random F1-score"])

            picket_pr.append(picket_evals[tomo_id]["Precision"])
            picket_re.append(picket_evals[tomo_id]["Recall"])
            picket_f1.append(picket_evals[tomo_id]["F1-score"])

            picket_rpr.append(picket_evals[tomo_id]["Random Precision"])
            picket_rre.append(picket_evals[tomo_id]["Random Recall"])
            picket_rf1.append(picket_evals[tomo_id]["Random F1-score"])
    all_milopyp_pr.append(milopyp_pr)
    all_milopyp_re.append(milopyp_re)
    all_milopyp_f1.append(milopyp_f1)
    all_milopyp_rpr.append(milopyp_rpr)
    all_milopyp_rre.append(milopyp_rre)
    all_milopyp_rf1.append(milopyp_rf1)
    all_picket_pr.append(picket_pr)
    all_picket_re.append(picket_re)
    all_picket_f1.append(picket_f1)
    all_picket_rpr.append(picket_rpr)
    all_picket_rre.append(picket_rre)
    all_picket_rf1.append(picket_rf1)


plt.savefig(
    f"workflow_comparisions/comparison_w_milopyp_tomotwin_11tomo_allrounds_denoised.png"
)
