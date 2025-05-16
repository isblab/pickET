# Understanding the outputs from PickET:

Standard outputs.  
To convert it to just centroids, use this script. 
Optionally to get 
Use script to get XYZ in CSV instead of YAML

You can see YAML, XYZ, see_centroids, and also the instance segmentations. 

1. Script to get XYZ in CSV instead of YAML
2. Flag for saving instance segmentations. 
3. see_centroids to get centroids overlaid on segmentation



# Output for S1

### Visualizing segmentations from S1: <a name="vis_seg_s1"></a>

***Note:*** *This step launches the [Napari](https://napari.org/) GUI. If the PickET workflow is being run on a remote computing node, we recommend users to connect to the remote computing node using `ssh` with `-X` flag. In other words, connect to the remote computing node using the following command:*  

```
ssh -X <username>@<ip_address of the remote computing node>
```

<br/>

Now, run the following command on each of the segmentations to visualize an overlay of the segmentation on the input tomogram in Napari:  
```
python see_segmentations.py <path_to_segmentation>
```



#### Tips to identify suitable segmentations for the second step

Look at the each of the six segmentations generated from S1 for the tomograms in the dataset. From all the segmentations generated from S1 for a given tomogram, identify the segmentation(s) in which particles are well separated from the background. More than one segmentation may be chosen for the next step. Also, note the voxel value for the voxel corresponding to particles in the segmentation. This could be 0 or 1 and is called the `particle_cluster_id` for each of the chosen segmentations. Either of the following ways can be used to obtain the `particle_cluster_id`:

1. Hover over a target particle in the loaded Napari window. This number should appear at the bottom of the window next to the coordinates of the mouse pointer.

2. Check if a target particle is colored with a non-gray color in this overlay. If the particle is colored, then `particle_cluster_id = 1`, else `particle_cluster_id = 0`.

*Note that the `particle_cluster_id` may not be the same for all the segmentations generated from S1 for a given tomogram.*

<br/>


# Output for S2










