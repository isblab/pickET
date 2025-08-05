import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"


def load_yaml(fname: str):
    particle_ids = []
    recalls = []
    with open(fname, "r") as infl:
        entries = yaml.safe_load(infl)

    for k, v in entries.items():
        particle_ids.append(k)
        recalls.append(v["Average recall"])
    return particle_ids, recalls


def main():
    picket_particle_wise_recalls_fname = sys.argv[1]
    milopyp_particle_wise_recalls_fname = sys.argv[2]

    picket_pids, picket_recalls = load_yaml(picket_particle_wise_recalls_fname)
    milopyp_pids, milopyp_recalls = load_yaml(milopyp_particle_wise_recalls_fname)
    assert picket_pids == milopyp_pids  # Break if the order is inconsistent

    df1 = pd.DataFrame(
        {
            "milopyp": milopyp_recalls,
            "picket": picket_recalls,
            "picket_pids": picket_pids,
        }
    )

    xys = np.linspace(0, 1, 20)
    df2 = pd.DataFrame({"x": xys, "y": xys})

    fig = px.scatter(
        df1,
        x="milopyp",
        y="picket",
        title="Comparison of particle-wise recalls between MiLoPYP and PickET",
        labels={"milopyp": "MiLoPYP recall", "picket": "PickET recall"},
        hover_data=["milopyp", "picket", "picket_pids"],
    )
    fig.add_trace(
        go.Scatter(
            x=df2["x"],
            y=df2["y"],
        )
    )
    fig.show()

    plt.scatter(df1["milopyp"], df1["picket"], s=5, c="Orange")
    plt.plot(df2["x"], df2["y"], color="#aeaeae", alpha=0.7, linestyle="--")
    plt.xlabel("Recall on MiloPYP predictions")
    plt.ylabel("Recall on PickET predictions")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(
        "/home/shreyas/Dropbox/miningTomograms/comparison_w_milopyp/particlewise_recall_comparison_w_milopyp.png",
        dpi=600,
    )


if __name__ == "__main__":
    main()
