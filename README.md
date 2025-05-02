<a href="https://pubmed.ncbi.nlm.nih.gov/38391029/">
    <img src="https://cdn.ncbi.nlm.nih.gov/pubmed/7e7ad262-b370-490b-9751-e68ac1c1c5b4/core/images/pubmed-logo-white.svg" alt="PubMed" width="150"/>
</a>

[![DOI](https://zenodo.org/badge/DOI/)](https://doi.org/) #TODO: Update Zenodo badge

# **PickET:** Unsupervised particle picking protocol for cryo-electron tomograms
Python package to pick particles in cryo-electron tomograms in an unsupervised manner


<img src="images/PickET_GraphicalAbstract.png" alt="PickET graphical abstract" width="600">

### **A schematic description of the PickET workflow:**
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
    A. Check the CUDA version by running `nvidia-smi`
    B. If the CUDA version is 12.\<something\>, install CuPy by running `pip install cupy-cuda12x`. Else if CUDA version is 11.\<something\>, install CuPy by running `pip install cupy-cuda11x`.

## Particle picking using PickET
1. Generate segmentations by running:
```
python s1.py <param_file_path>
```

## !TODO
---
### Inputs

(See also `examples/`)
...

### Run PickET
...

## Outputs

### Plots

### Output YAML file

## Choice of parameters


## Information
__Author(s):__ Shreyas Arvindekar, Shruthi Viswanath

__Date__: June 1st, 2025

__License:__ [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License.

__Testable:__ Yes

__Parallelizeable:__ Yes

__Publications:__  Arvindekar, S., _et. al._ PickET - A library of methods for unsupervised particle-picking from cryo-electron tomograms, . #TODO: Add a formal citation