# Step 2 (S2): Localize particles  
## Inputs for S2:

Inputs for S1 are provided through a YAML file containing parameters. An example is provided in `examples/s1_params_example.yaml`. These parameters are described in detail below:

```yaml
dataset_name: <An identifier for the dataset>
```

`dataset_name` is also the name of the directory where all the outputs will be saved.
```yaml
inputs: 
[  
    {
        segmentation: <path_to_segmentation_1>, 
        particle_cluster_id: 1,  # Cluster index of the particle cluster 
        lower_z-slice_limit: <upper_zslice_where_the_lamella_starts>, #[Optional]#
        upper_z-slice_limit: <lower_zslice_where_the_lamella_ends> #[Optional]#
        },
    {
        segmentation: <path_to_segmentation_1>, 
        particle_cluster_id: 0,
        lower_z-slice_limit: <upper_zslice_where_the_lamella_starts>, #[Optional]#
        upper_z-slice_limit: <lower_zslice_where_the_lamella_ends> #[Optional]#
        },
]
```

`inputs` is a list (enclosed within square brackets) that can be expanded with similar entries, enclosed in curly brackets as shown above.  
<br/>

`segmentation` and the corresponding `particle_cluster_id` are obtained from the [visualizing segmentations](visualizing_segmentations.md). 
`lower_z-slice_limit` and `upper_z-slice_limit` denote the upper and lower bounds on the Z-slices where the tomogram is likely to contain particles. 

!!! note "Note" 
    *These bounds define the bounds on the region from which particles will be picked. These bounds can be more relaxed than the ones used for [generating semantic segmentations](running_s1.md). See also Fig 2.* 

<div align="center">
    <img src="../images/Zbounds.jpg" alt="Fig. 2: Z-slice bounds for the two steps in PickET" width="600" align="center">
    <p align="center"><b>Fig. 2:</b> Z-slice bounds for the two steps in PickET </p>
</div>  

```yaml
particle_extraction_params: 
[
    {mode: connected_component_labeling},
    {mode: watershed_segmentation, min_distance: 15}
]
```

Similar to the `inputs`, `particle_extraction_params` is also a list of dictionaries. Each dictionary defined in this list defines a particle extraction mode. Here, we provide two particle extraction modes `connected_component_labeling` and `watershed_segmentation`. 

First, for `mode: connected_component_labeling`, there are no hyperparameters. This mode is fast and works well for less crowded datasets.

Second, for `mode: watershed_segmentation`, there is one hyperparameter. This mode uses watershed segmentation for converting the semantic segmentation into instance segmentation. It uses the `min_distance` hyperparameter that defines the minimum separation between two detected particles in voxels.

```yaml
output_dir: /data/picket_results/
```

As the name suggests, `output_dir` describes the path to the directory where the output segmentations will be saved.  
!!! note "Note"
    *The extracted particle centroid coordinates will be saved as `.yaml` files in `output_dir/dataset_name/predicted_particles/` directory.*

<br/>

---
<br/>

[Back to Home](index.md)  
[Go to usage instructions](usage_instructions.md)  
[Go to running s2](running_s2.md)
