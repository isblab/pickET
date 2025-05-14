# Getting started
This document provides a detailed guide for installing PickET.

## Dependencies:
* Python dependencies are mentioned in `requirements.txt`.
* CuPy
* TomoEED
Instructions for installing the Python dependencies, CuPy and TomoEED are provided below:

## Installation:
### 1. Install all the Python dependencies:
```
pip install -r requirements.txt
```

### 2. Install CuPy:  
**If the CUDA version is `12.<something>`, install CuPy by running**  
    
```
pip install cupy-cuda12x
```   
**Else if CUDA version is `11.<something>`, install CuPy by running.**  
```
pip install cupy-cuda11x
```
**_Note:_** The CUDA version can be checked by running `nvidia-smi` from the terminal.

### 3. Install TomoEED:  
1. Visit the official [TomoEED webpage](https://sites.google.com/site/3demimageprocessing/tomoeed).  

2. Fill and submit the Google Form.  

3. Copy the generated download link and paste it in a new browser tab. This will download TomoEED as a zipped directory. Unzip it.   
   

---
***Note:***  
TomoEED can be run by running the following command:
```
<path_to_the_unzipped_directory>/bin/tomoeed path_to_input_tomogram/input_tomogram.mrc denoised_tomograms/output_tomogram.mrc
```
This will make a denoised version of the input tomogram (`path_to_input_tomogram/input_tomogram.mrc`) and place it at `denoised_tomograms/output_tomogram.mrc`.


[Back to Home](README.md)  
[Go to usage instructions](run_picket.md)
