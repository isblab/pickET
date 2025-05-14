# Step 1 (S1): Generate semantic segmentations 
## Running S1:

### 1. Denoise tomograms:
First, denoise the input tomograms using `TomoEED` by running the following command:
```
<path_to_the_unzipped_tomoeed_directory>/bin/tomoeed path_to_input_tomogram/input_tomogram.mrc denoised_tomograms/output_tomogram.mrc
```
This will make a denoised version of the input tomogram (`path_to_input_tomogram/input_tomogram.mrc`) and place it at `denoised_tomograms/output_tomogram.mrc`.  

***Note:*** *Make sure to use the path to denoised tomogram (`denoised_tomograms/output_tomogram.mrc`) in the `s1_param_file_path` for `tomogram` in the `inputs` section.*  

<br/>

### 2. Generate semantic segmentations
Run the following command to generate the six semantic segmentations.
```
python s1.py <s1_param_file_path>
```

<br/>

---
<br/>

[Back to Home](../README.md)  
[Go to outputs](outputs.md)  
[Go to usage instructions](usage_instructions.md)  

