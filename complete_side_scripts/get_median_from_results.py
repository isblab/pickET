import sys
import yaml
import numpy as np


def read_results_file(result_fname: str):
    with open(result_fname, "r") as f:
        results = yaml.safe_load(f)
    return results


def get_metric_values(results: dict[str, dict[str, float]], metric: str) -> np.ndarray:
    metric_values = []
    for k, v in results.items():
        if k.startswith("Tomo"):
            metric_values.append(v[metric])
    metric_values = np.array(metric_values)
    return metric_values


def main():
    result_fname = sys.argv[1]
    metric = sys.argv[2]

    results = read_results_file(result_fname)
    results_metric_values = get_metric_values(results, metric)
    print(f"Captured {len(results_metric_values)} values...")

    print(
        f"Mean: {np.mean(results_metric_values):.4f}\t\tMedian: {np.median(results_metric_values):.4f}"
    )


if __name__ == "__main__":
    main()
