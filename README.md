<a href="https://pubmed.ncbi.nlm.nih.gov/38391029/">
    <img src="https://cdn.ncbi.nlm.nih.gov/pubmed/7e7ad262-b370-490b-9751-e68ac1c1c5b4/core/images/pubmed-logo-white.svg" alt="PubMed" width="150"/>
</a>

[![DOI](https://zenodo.org/badge/DOI/)](https://doi.org/) #TODO: Update Zenodo badge

# **PickET:** Unsupervised particle picking protocol for cryo-electron tomograms
Python package to pick particles in cryo-electron tomograms in an unsupervised manner

<img src="images/PickET_AlgorithmFlowchart.png" alt="PickET workflow" width="600">


## Publication and Data
* Shreyas Arvindekar, Omkar Golatkar and Shruthi Viswanath, **PickET - A library of methods for unsupervised particle-picking from cryo-electron tomograms**, . #TODO: Add a formal citation
* Data is deposited in [Zenodo](https://www.doi.org/) #TODO: Update Zenodo ID


## Dependencies:
* See `requirements.txt` for Python dependencies

## Installation:
1. Install all the dependencies:
```
pip install -r requirements.txt
```
2. Install CuPy:  
    **If the CUDA version is `12.<something>`, install CuPy by running**  
    ```
    pip install cupy-cuda12x
    ```   
    **Else if CUDA version is `11.<something>`, install CuPy by running.**  
    ```
    pip install cupy-cuda11x
    ```
    **_Note:_** The CUDA version can be checked by running `nvidia-smi` from the terminal.
    


## Run PickET
The workflow for running PickET is split into two steps - S1 and S2. The first step involves generating semantic segmentations for each of the input tomograms. By default, the first step of PickET uses all three feature extraction modes along with both clustering workflows to generate six semantic segmentations for each input tomogram. The users can then choose to proceed with using one or more of these segmentations for particle extraction and other downstream workflows. All the inputs for PickET are stored in two separate `param_file.yaml` files, namely `s1_params.yaml` and `s2_params.yaml` (See also `examples/`). The parameters in this file are described below:

### Inputs for S1:
An example set of params for S1 are provided in `examples/s1_params_example.yaml`. These parameters are described in detail below:

    dataset_name: <An identifier for the dataset>
This will also be the name of the directory where all the outputs will be saved.

    inputs: 
    [  
        {
        tomogram: <path_to_tomogram_1>,
        lower_z-slice_limit: <upper_zslice_where_the_lamella_starts>, #[Optional]#
        upper_z-slice_limit: <lower_zslice_where_the_lamella_ends> #[Optional]#
            },
        {
        tomogram: <path_to_tomogram_2>,
        lower_z-slice_limit: <upper_zslice_where_the_lamella_starts>, #[Optional]#
        upper_z-slice_limit: <lower_zslice_where_the_lamella_ends> #[Optional]#
            },
    ]
This list can be expanded with similar entries, enclosed in curly brackets as shown above. If the user does not want to specify the entries marked as `#[Optional]#` in the `input` section shown above, they should delete these lines from the `param_file.yaml`.

    window_size: 5
We recommend using the $window\_size: 5$ for picking particles from tomograms.

    max_num_windows_for_fitting: 100_000_000
This parameter specifies the number of voxel neighborhoods to be used in the first pass to fit the clustering algorithm. Reducing this number will reduce the computational power required, but will come at the cost of performance. On the contrary, increasing this number might require more computing power and may result in better segmentations. The users can see the number of neighborhoods being used in a given run by referring to the terminal output for the run. It will be shown as `Features array of shape: (<num_neighborhoods_being_used>, 125)` in the output.

We recommend users to optimize this number according to the computing power available. The users will need to optimize this number only once for their computing system. Once optimized, the same can be used for all datasets that will be processed using PickET on that computing node in the future.

    feature_extraction_params: 
    [
        {
        mode: ffts, 
        n_fft_subsets: 64,
            },
        
        {
        mode: gabor, 
        num_sinusoids: 10, 
        num_windows_subsets: 5,
        num_parallel_filters: 8,
        num_output_features: 64
            },
        
        {
        mode: intensities
            }  
    ]
These hyperparameters describe the feature extraction process. `feature_extraction_params` is also a list of dictionaries, similar to the `inputs`. Each dictionary defined in this list defines a feature extraction mode. Here, we provide three feature extraction modes `ffts`, `gabor` and `intensities`. 

First, for `mode: ffts`, there is only one hyperparameter, `n_fft_subsets`. This hyperparameter defines how many neighborhoods will be processed simultaneously for feature extraction. Higher the value, the faster the process will run, but will require more computational power.

Second, for `mode: gabor`, there are four key hyperparameter. The size of the Gabor filter bank is the cube of the `num_sinusoids`. The user may choose to not tweak this hyperparameter. The `num_windows_subsets` and `num_parallel_filters` define the number of windows and number of Gabor filters to be processed simultaneously. Increasing the `num_windows_subsets` and reducing the `num_parallel_filters` will result in the feature extraction requiring less computing power, but will take longer to process the tomogram. The `num_output_features` defines the number of features with the highest standard deviation to be used for clustering. The user may choose not to tweak this hyperparameter.

Third, for `mode: intensities`, there are no hyperparameters. It will use the voxel intensities obtained from the neighborhoods as features for clustering.


    clustering_methods: [kmeans, gmm]
The `clustering_methods` list described the clustering algorithms to be used. In this example, both `KMeans` as well as `GMM` will be used for clustering.

    output_dir: /data/picket_results/
As the name suggests, `output_dir` describes the path to the directory where the output segmentations will be saved.  
*Note:* The segmentations will be saved in `output_dir/dataset_name` directory.

---

### Inputs for S2:
An example set of params for S1 are provided in `examples/s1_params_example.yaml`. These parameters are described in detail below:

    dataset_name: <An identifier for the dataset>
This will also be the name of the directory where all the outputs will be saved.

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
*Note:* The `particle_cluster_id` is essentially the voxel intensity for the voxel corresponding to particles voxel in the segmentation.

This list can be expanded with similar entries, enclosed in curly brackets as shown above.

    particle_extraction_params: 
    [
        {mode: connected_component_labeling},
        {mode: watershed_segmentation, min_distance: 15}
    ]
These hyperparameters define the particle extraction workflow. Similar to the `inputs`, `particle_extraction_params` is also a list of dictionaries. Each dictionary defined in this list defines a particle extraction mode. Here, we provide two particle extraction modes `connected_component_labeling` and `watershed_segmentation`. 

First, for `mode: connected_component_labeling`, there are no hyperparameters. This mode is fast and works well for less crowded datasets.

Second, for `mode: watershed_segmentation`, there is one hyperparameter. This mode uses the watershed segmentation workflow for splitting semantic segmentation into instance segmentation. It uses the `min_distance` hyperparameter that defines the minimum separation between two detected particles in voxels.

    output_dir: /data/picket_results/
As the name suggests, `output_dir` describes the path to the directory where the output segmentations will be saved.  
*Note:* The extracted particle centroid coordinates will be saved as `.ndjson` files in `output_dir/dataset_name/predicted_particles/` directory.

---

## Particle picking using PickET
### Step 1 (S1): Generate semantic segmentations:
First, run the following command to generate the semantic segmentations.
```
python s1.py <s1_param_file_path>
```
Second, run the following command on each of the segmentation to visualize an overlay of the segmentation on the input tomogram  in Napari:
```
python see_segmentations.py <path_to_segmentation>
```
From this step, the users will also get the `particle_cluster_id` for each of the segmentation, which will be needed for S2.


### Step 2 (S2): Extract particle centroids by running:
Run the following command to obtain centroids for predicted particles.
```
python s2.py <s2_param_file_path>
```

## Outputs
First, the output from S1 are the semantic segmentations in `.h5` file format. An overlay of these on the input tomogram can be visualized in Napari by running `python see_segmentations.py <path_to_segmentation>`.


## Information
__Author(s):__ Shreyas Arvindekar, Shruthi Viswanath  
__Date__: June 1st, 2025  
__License:__ [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License.  
__Testable:__ Yes  
__Publications:__  Arvindekar, S., _et. al._ PickET - A library of methods for unsupervised particle-picking from cryo-electron tomograms, . #TODO: Add a formal citation