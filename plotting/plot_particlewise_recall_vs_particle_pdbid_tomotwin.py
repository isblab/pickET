# import os
import sys
import yaml
import numpy as np

# import matplotlib.pyplot as plt
import pandas as pd
import plotly.io as pio
import plotly.express as px

pio.renderers.default = "browser"


def main():
    particlewise_recall_fname = sys.argv[1]

    with open(particlewise_recall_fname, "r") as input_f:
        particlewise_recall = yaml.safe_load(input_f)

    all_average_recalls, all_particle_ids = [], []
    for particle_id, results in particlewise_recall.items():
        all_average_recalls.append(results["Average recall"])
        all_particle_ids.append(particle_id)

        # recalls = results["Recalls"]
        # plt.scatter(
        #     x=recalls,
        #     y=[particle_id for _ in recalls],
        #     color="Orange",
        #     alpha=0.75,
        #     zorder=2,
        # )

    all_average_recalls, all_particle_ids = (
        np.array(all_average_recalls),
        np.array(all_particle_ids),
    )
    print(all_average_recalls.shape, all_particle_ids.shape)
    # plt.barh(y=all_particle_ids, width=all_average_recalls, alpha=0.75, zorder=1)
    df = pd.DataFrame({"x_values": all_average_recalls, "y_values": all_particle_ids})
    fig = px.scatter(
        df,
        x="x_values",
        y="y_values",
        title="Interactive Scatter Plot (Plotly)",
        labels={"x_values": "Recall", "y_values": "PDB ID"},
        hover_data=["x_values", "y_values"],
    )
    fig.update_yaxes(
        tickmode="array",  # Use an array of values to define ticks
        tickvals=df["y_values"],  # Set the ticks to be at each category's position
        ticktext=df["y_values"],  # Set the label for each tick
        showgrid=True,
        gridwidth=1,
        gridcolor="LightGrey",
    )
    fig.show()

    # fname_head = particlewise_recall_fname.split("/")[-1][:-5]
    # feature_extraction_method = fname_head.split("_")[2]
    # clustering_method = fname_head.split("_")[3]
    # particle_extraction_method = "_".join(fname_head.split("_")[4:])
    # plt.savefig(
    #     os.path.join(
    #         *particlewise_recall_fname.split("/")[:-1],
    #         f"particlewise_recall_{feature_extraction_method}_{clustering_method}_{particle_extraction_method}.png",
    #     ),
    #     dpi=600,
    # )
    # plt.show()
    # plt.close()


if __name__ == "__main__":
    main()
