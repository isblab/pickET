# Getting started
This document provides a detailed guide for installing PickET.

## Dependencies

* Python dependencies can be found in `pyproject.toml`.
* [TomoEED](https://sites.google.com/site/3demimageprocessing/tomoeed)
  
Instructions for installing the Python dependencies, CuPy and TomoEED are provided below:

## Installation
### 1. Create a new virtual environment

In a new terminal window, run the following command to install venv package:
```bash 
pip install venv
```
!!! note
    *On Ubuntu-like operating systems, this command needs to be run as shown here:*
    ```bash 
    sudo apt install python3-venv 
    ```

Then, run the following commands sequentially to create a new virtual environment:
```bash
mkdir <path_to_new_virtual_environment>
cd <path_to_new_virtual_environment>
python -m venv picket_env
```


### 2. Activate the environment <a name="env_activate"></a>
```bash
source <path_to_new_virtual_environment>/bin/activate
```
!!! note 
    *Make sure to activate the environment before installing rest of the dependencies and before running PickET. The virtual environment can be deactivated by running `deactivate` in the terminal or simply by closing the terminal window.*



### 3. Install [CuPy](https://cupy.dev)    

If the CUDA version is `12.<something>`, install CuPy by running
    
```
pip install cupy-cuda12x
```

Else if CUDA version is `11.<something>`, install CuPy by running.

```
pip install cupy-cuda11x
```

!!! note
    *The CUDA version can be checked by running `nvidia-smi` from the terminal. It will be shown on the top right corner in the generated output.*



### 4. Install all the PickET

```bash
git clone https://github.com/isblab/pickET.git
cd pickET
pip install .
```
Installing PickET via `pip` using the above command builds executables for the two steps in a PickET run - generating semantic segmentation and localizing particles. (See also [Usage instructions](usage_instructions.md))


### 5. Install TomoEED  
1. Visit the official [TomoEED webpage](https://sites.google.com/site/3demimageprocessing/tomoeed).  

2. Fill and submit the Google Form.  

3. Copy the generated download link and paste it in a new browser tab. This will download TomoEED as a zipped directory. Unzip it.   
   
---

[Back to Home](index.md)  
[Go to usage instructions](usage_instructions.md)  
