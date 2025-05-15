# Usage instructions

PickET is a modular library that provides a variety of methods for particle-picking in cryo-electron tomograms. 

The workflow for running PickET is split into two steps - S1 and S2, described in more details below. 

***Note:*** *We strongly recommend running all the steps in the PickET workflow on a single computing node (a local workstation or a remote computing cluster). Several intermediate files are generated at different stages in the workflow. These files contain pointers to the data required for downstream processes. These pointers may not work if the data is transferred to a different computing system.*

***Important:*** *Make sure to activate the environment before running PickET. See [create and activate virtual environment](installation.md#env_activate) for more details.*

## S1 - Generate semantic segmentation

The first step (S1, semantic segmentation) identifies voxels corresponding to particles in each input tomogram. This step involves three feature extraction modes (`FFTs`, `Gabor` and `intensities`) and two clustering methods (`KMeans` and `GMM`) to classify each voxel as particle or background. In total, this generates six semantic segmentations for each input tomogram, corresponding to every combination of feature extraction mode and clustering method. The users may then proceed with one or more of these six segmentations for the second step. 

*Note that a workflow that generates the most optimal segmentation for a given tomogram may not necessarily generate the most optimal segmentations for all tomograms in that dataset.*  


#### [**Inputs for S1**](input_for_s1.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[**How to run S1?**](running_s1.md)

---
<br/>

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

1. Hover over a target particle in the loaded Napari window. #TODO UNCLEAR with the next step.... This number should appear at the bottom of the window next to the coordinates of the mouse pointer

2. Check if a target particle is colored with a non-gray color in this overlay. If the particle is colored, then `particle_cluster_id = 1`, else `particle_cluster_id = 0`.

*Note that the `particle_cluster_id` may not be the same for all the segmentations generated from S1 for a given tomogram.*

<br/>

## S2 - Extract particles 

In the second step (S2, particle extraction), particles segmentations are obtained using two segmentation methods (`connected component labeling` and `watershed segmentation`), allowing the user to choose between the two. The centroids of predicted particles are provided as output. Users also have an option to obtain subtomograms enclosing the predicted particles for downstream subtomogram averaging. #TODO there are more outputs we get. #TODO why are we describing all outputs here? #TODO need a separate tab for visualizing output of S2.  

#### [**Inputs for S2**](input_for_s2.md)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[**How to run S2?**](running_s2.md)

---
<br/>

[Back to Home](README.md)  
[Go to installation instructions](installation.md)
