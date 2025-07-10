# Step 1 (S1): Generate semantic segmentations 
## Running S1

### 1. Denoise tomograms
First, denoise the input tomograms using `TomoEED` by running the following command:
```bash
<tomoeed_directory>/bin/tomoeed full_path_to_input_tomogram.mrc denoised_tomograms/output_tomogram.mrc
```
This will make a denoised version of the input tomogram (`input_tomogram.mrc`) and place it at `denoised_tomograms/output_tomogram.mrc`. If running from a different directory than `input_tomogram.mrc`, you will need to specify the full path to the input tomogram.

!!! warning "Important"
    *Make sure to use the path to denoised tomogram (`denoised_tomograms/output_tomogram.mrc`) in the `s1_param_file_path` for `tomogram` in the `inputs` section.*  


### 2. Generate semantic segmentations
Run the following command to generate the semantic segmentations.
```bash
python s1.py <s1_param_file_path>
```
Next step is to choose one or more optimal segmentations for each input tomogram and obtaining the corresponding particle cluster IDs. Follow the instructions at [visualizing segmentations](visualizing_segmentations.md) and [obtaining particle cluster ID](obtaining_particle_cluster_id.md) for more details on this.

---
<br/>

[Back to Home](index.md)  
[Go to outputs](outputs.md)  
[Go to usage instructions](usage_instructions.md)  

