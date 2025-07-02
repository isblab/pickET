# Preprocessing the ground truth annotations 
## Offset correction and concatenating the annotation files
We use ground truth files in `.ndjson` format with the particle centroid coordinates mentioned with respect to the origin situated at the top left front corner of the tomogram. 

### For TomoTwin dataset
We noticed that the ground truth annotations had an offset. This was corrected by running the [tomotwin ground truth preprocessing script](https://github.com/isblab/pickET/blob/main/accessories/preprocess_ground_truth/generate_ndjson_ground_truth_tomotwin.py) which can be run as follows:
```bash
python accessories/generate_ndjson_ground_truth_tomotwin.py <parent_dir> <x> <y> <z>
```

where, `parent_dir` is the directory of a round of simulated tomograms, and `x`, `y` and `z` denote the size of the tomogram along the three axes.

### For CZI datasets
We noticed that the ground truth annotations did not have any offset. The coordinate files were concatenated to a single file by [czi ground truth preprocessing script](https://github.com/isblab/pickET/blob/main/accessories/preprocess_ground_truth/generate_ndjson_ground_truth_czi.py) which can be run as follows:
```bash
python accessories/generate_ndjson_ground_truth_czi.py <parent_dir> 
```

where `<parent_dir>` corresponds to the `annotations` directory associated with each tomogram.
