import sys
import yaml
import numpy as np


def collect_metric_values(
    results: dict[str, dict[str, float]], metric: str
) -> list[float]:
    vals = []
    for tomo, res in results.items():
        if tomo.startswith("Tomo"):
            vals.append(res[metric])
    return vals


def main():
    respaths = sys.argv[1:]
    precisions, recalls, f1scores = [], [], []
    mdr_recalls, relative_recalls = [], []
    for respath in respaths:
        with open(respath, "r") as f:
            results = yaml.safe_load(f)
            precisions.extend(collect_metric_values(results, "Precision"))
            recalls.extend(collect_metric_values(results, "Recall"))
            f1scores.extend(collect_metric_values(results, "F1-score"))
            mdr_recalls.extend(collect_metric_values(results, "MDR Recall"))
            relative_recalls.extend(collect_metric_values(results, "Relative recall"))

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1scores = np.array(f1scores)
    mdr_recalls = np.array(mdr_recalls)
    relative_recalls = np.array(relative_recalls)
    oneminus_mdr = 1 - mdr_recalls

    with open("metric_vals.csv", "w") as outf:
        outf.write(
            "precision,recalls,f1scores,mdr_recalls,oneminus_mdrrecalls,relative_recall\n"
        )
        for i in range(len(precisions)):
            outf.write(
                f"{precisions[i]},{recalls[i]},{f1scores[i]},{mdr_recalls[i]},{oneminus_mdr[i]},{relative_recalls[i]}\n"
            )


if __name__ == "__main__":
    main()
