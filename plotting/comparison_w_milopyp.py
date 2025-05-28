import sys
import yaml
import matplotlib.pyplot as plt

milopyp_eval_path = sys.argv[1]
picket_eval_path = sys.argv[2]

with open(milopyp_eval_path, "r") as mep:
    milopyp_evals = yaml.safe_load(mep)

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

        tomo_id = f"Tomo_ID - {int(tomo_id.split(' - ')[-1])*8}"  #! Remove this line if not using the tomo11 run

        picket_pr.append(picket_evals[tomo_id]["Precision"])
        picket_re.append(picket_evals[tomo_id]["Recall"])
        picket_f1.append(picket_evals[tomo_id]["F1-score"])

        picket_rpr.append(picket_evals[tomo_id]["Random Precision"])
        picket_rre.append(picket_evals[tomo_id]["Random Recall"])
        picket_rf1.append(picket_evals[tomo_id]["Random F1-score"])

fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].scatter(x=["MiLoPYP" for _ in milopyp_pr], y=milopyp_pr, color="red", alpha=0.7)
ax[0].scatter(
    x=["MiLoPYP" for _ in milopyp_rpr], y=milopyp_rpr, color="#aeaeae", alpha=0.7
)
ax[0].scatter(x=["PickET" for _ in picket_pr], y=picket_pr, color="#00bfff", alpha=0.7)
ax[0].scatter(
    x=["PickET" for _ in picket_rpr], y=picket_rpr, color="#aeaeae", alpha=0.7
)
ax[0].set_ylabel("Precision")

ax[1].scatter(x=["MiLoPYP" for _ in milopyp_re], y=milopyp_re, color="red", alpha=0.7)
ax[1].scatter(
    x=["MiLoPYP" for _ in milopyp_rre], y=milopyp_rre, color="#aeaeae", alpha=0.7
)
ax[1].scatter(x=["PickET" for _ in picket_re], y=picket_re, color="#00bfff", alpha=0.7)
ax[1].scatter(
    x=["PickET" for _ in picket_rre], y=picket_rre, color="#aeaeae", alpha=0.7
)
ax[1].set_ylabel("Recall")

ax[2].scatter(x=["MiLoPYP" for _ in milopyp_f1], y=milopyp_f1, color="red", alpha=0.7)
ax[2].scatter(
    x=["MiLoPYP" for _ in milopyp_rf1], y=milopyp_rf1, color="#aeaeae", alpha=0.7
)
ax[2].scatter(x=["PickET" for _ in picket_f1], y=picket_f1, color="#00bfff", alpha=0.7)
ax[2].scatter(
    x=["PickET" for _ in picket_rf1], y=picket_rf1, color="#aeaeae", alpha=0.7
)
ax[2].set_ylabel("F1-score")

fig.set_figheight(10)

plt.savefig(
    f"workflow_comparisions/comparison_w_milopyp_tomotwin_11tomo_allrounds_denoised.png"
)
