# Usage instructions

PickET is a modular library that provides a variety of workflows for particle-picking in cryo-electron tomograms. 

A typical PickET run comprises two steps - S1 and S2, described in more detail below. 

!!! note 
    *We strongly recommend running all the steps described below on a single computing node (a local workstation or a remote computing cluster). Several intermediate files are generated at different stages in the pipeline. These files contain pointers to the input data which is required for downstream processes. These pointers may not work if the data is transferred to a different computing system.*

!!! warning 
    *Make sure to activate the environment before running PickET. See [create and activate virtual environment](installation.md#env_activate) for more details.*

## S1 - Generate semantic segmentation

The first step (S1, semantic segmentation) identifies voxels corresponding to particles in each input tomogram. This step involves three feature extraction modes (`FFTs`, `Gabor`, and `intensities`) and two clustering algorithms (`KMeans` and `GMM`) to classify each voxel as a particle or background. In total, this generates six semantic segmentations for each input tomogram, corresponding to every combination of feature extraction mode and clustering algorithm. The users may then proceed with one or more of these six segmentations for the second step. 

!!! note
    *That a workflow that generates the most optimal segmentation for a given tomogram may not necessarily generate the most optimal segmentations for all tomograms in that dataset.*  

The output segmentations generated from S1 can be visualized by following the instructions in [visualizing the output segmentations](visualizing_segmentations.md). From all the segmentations generated from S1 for a given tomogram, identify the segmentation(s) in which particles are well separated from the background. More than one segmentation may be chosen for the next step. Follow the instructions in [obtaining particle cluster ID](obtaining_particle_cluster_id.md) to get the voxel value for the voxel corresponding to particles in the segmentation. This value is specific for each segmentation and is passed as an input (`particle_cluster_id`) for S2.


[**Inputs for S1**](input_for_s1.md)&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;[**How to run S1?**](running_s1.md)&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;[**Outputs**](outputs.md)  
[**Visualizing the output segmentations**](visualizing_segmentations.md)&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;[**Obtaining particle cluster ID**](obtaining_particle_cluster_id.md)  


## S2 - Localize particles

In the second step (S2, particle localization), particle segmentations are obtained using two segmentation methods (`connected component labeling` and `watershed segmentation`), allowing the user to choose between the two. The centroids of predicted particles are provided as output. Users also have the option to obtain subtomograms enclosing the predicted particles for downstream subtomogram averaging. 

[**Inputs for S2**](input_for_s2.md)&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;[**How to run S2?**](running_s2.md)&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;[**Outputs**](outputs.md)  
[**Visualizing the output segmentations**](visualizing_segmentations.md)


!!! note 
    *PickET is best run on a large computing node (a local workstation or a remote computing cluster, if you have access to high-performance computing). Otherwise, the following options can be adjusted to run it on the available memory.  
       1. The central Z-slab can be made narrower for fitting the clusterer by setting a `lower_z-slice_limit` and `upper_z-slice_limit`.
       2. The `max_num_neighborhoods_for_fitting` can be decreased. 
    Note that these changes need to be made only for the S1 stage (feature extraction and clustering) and not S2 stage.*

<br/>

---

[Back to Home](index.md)  
[Go to installation instructions](installation.md)
