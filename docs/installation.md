# Getting started
This document provides a detailed guide for installing PickET.

## Dependencies:
* Python dependencies are mentioned in `requirements.txt`.
* [CuPy](https://cupy.dev) 
* [TomoEED](https://sites.google.com/site/3demimageprocessing/tomoeed)
  
Instructions for installing the Python dependencies, CuPy and TomoEED are provided below:

## Installation:
### 1. Create a new virtual environment:

In a new terminal window, run the following command to install venv package:
```
pip install venv
```

Then, run the following commands sequentially to create a new virtual environment:
```
mkdir <path_to_new_virtual_environment>
cd <path_to_new_virtual_environment>
python -m venv picket_env
```

#### Activate the environment <a name="env_activate"></a>
```
source <path_to_new_virtual_environment>/bin/activate
```
***Note:*** *Make sure to activate the environment before installing rest of the dependencies and before running PickET. The virtual environment can be deactivated by running `deactivate` in the terminal or simply by closing the terminal window.*



### 2. Install all the Python dependencies:

```
pip install -r requirements.txt
```


### 3. Install CuPy:  
**If the CUDA version is `12.<something>`, install CuPy by running**  
    
```
pip install cupy-cuda12x
```

**Else if CUDA version is `11.<something>`, install CuPy by running.**  

```
pip install cupy-cuda11x
```

**_Note:_** The CUDA version can be checked by running `nvidia-smi` from the terminal. It will be shown on the top right corner in the generated output.

### 4. Install TomoEED:  
1. Visit the official [TomoEED webpage](https://sites.google.com/site/3demimageprocessing/tomoeed).  

2. Fill and submit the Google Form.  

3. Copy the generated download link and paste it in a new browser tab. This will download TomoEED as a zipped directory. Unzip it.   
   
---
***Note:***  
#TODO this should be before step S1 in the main usage page or in S1 page itself. A bit non-intuitive to get the first step of method in the installation doc. 

TomoEED can be run by running the following command:
```
<path_to_the_unzipped_tomoeed_directory>/bin/tomoeed path_to_input_tomogram/input_tomogram.mrc denoised_tomograms/output_tomogram.mrc
```

This will make a denoised version of the input tomogram (`path_to_input_tomogram/input_tomogram.mrc`) and place it at `denoised_tomograms/output_tomogram.mrc`.


[Back to Home](README.md)  
[Go to usage instructions](usage_instructions.md#usage-instructions)  
