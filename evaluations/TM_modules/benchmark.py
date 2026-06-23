import yaml
import pandas as pd
import os
import matplotlib.pyplot as plt


def load_metrics(yaml_file):

    if not os.path.exists(yaml_file):
        print(f"Missing metrics file: {yaml_file}")
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0, 
            "num_predictions": 0
        }

    with open(yaml_file) as f:
        data = yaml.safe_load(f)

    return {
        "precision":
            data["precision"],
        "recall":
            data["recall"],
        "f1_score":
            data["f1_score"],
        "num_predictions":
            data["num_predictions"]
    }


def build_benchmark_dataframe(dataset, results_dir):

    rows = []
    
    for tomo in dataset:
        basename = (
            os.path.splitext(
                os.path.basename(
                    tomo["path"]
                )
            )[0]
        )

        tomo_results_dir = os.path.join(results_dir, basename)
        baseline_yaml = os.path.join(tomo_results_dir, "baseline_evaluation.yaml")
        picket_yaml = os.path.join(tomo_results_dir, "picket_evaluation.yaml")
        baseline = load_metrics(baseline_yaml)
        picket = load_metrics(picket_yaml)

        rows.append({
            "tomogram":
                basename,
            "baseline_precision":
                baseline["precision"],
            "baseline_recall":
                baseline["recall"],
            "baseline_f1":
                baseline["f1_score"],
            "baseline_predictions":
                baseline["num_predictions"],
            "picket_precision":
                picket["precision"],
            "picket_recall":
                picket["recall"],
            "picket_f1":
                picket["f1_score"],
            "picket_predictions":
                picket["num_predictions"],
            "delta_precision":
                picket["precision"] - baseline["precision"],
            "delta_recall":
                picket["recall"] - baseline["recall"],
            "delta_f1":
                picket["f1_score"] - baseline["f1_score"]
        })

    return pd.DataFrame(rows)


def compute_summary_statistics(df):

    summary = {
        "baseline_precision_mean":
            df["baseline_precision"].mean(),
        "baseline_recall_mean":
            df["baseline_recall"].mean(),
        "baseline_f1_mean":
            df["baseline_f1"].mean(),
        "picket_precision_mean":
            df["picket_precision"].mean(),
        "picket_recall_mean":
            df["picket_recall"].mean(),
        "picket_f1_mean":
            df["picket_f1"].mean(),
        "delta_precision_mean":
            df["delta_precision"].mean(),
        "delta_recall_mean":
            df["delta_recall"].mean(),
        "delta_f1_mean":
            df["delta_f1"].mean()
    }

    return pd.DataFrame([summary])


def generate_violin_plots(df, output_dir):

    metrics = ["precision", "recall", "f1"]
    for metric in metrics:
        plt.figure()
        plt.violinplot([df[f"baseline_{metric}"], df[f"picket_{metric}"]])
        plt.xticks([1,2], ["Baseline", "PickET"])
        plt.ylabel(metric)
        plt.title(f"{metric.capitalize()} Distribution")
        plt.savefig(os.path.join(output_dir, f"{metric}_violin.png"))
        plt.close()


def generate_boxplots(df, output_dir):

    metrics = ["precision", "recall", "f1"]
    for metric in metrics:
        plt.figure()
        plt.boxplot([df[f"baseline_{metric}"], df[f"picket_{metric}"]])
        plt.xticks([1,2], ["Baseline", "PickET"])
        plt.ylabel(metric)
        plt.title(f"{metric.capitalize()} Distribution")
        plt.savefig(os.path.join(output_dir, f"{metric}_boxplot.png"))
        plt.close()
