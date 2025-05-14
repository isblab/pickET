# Step 2 (S2): Extract particles  
## Running S2:
Run the following command to obtain centroids for predicted particles.

```
python s2.py <s2_param_file_path> 
```

Second, run the following command on each of the segmentation to visualize an overlay of the segmentation on the input tomogram in Napari:
```
python see_segmentations.py <path_to_segmentation>
```
An overlay of these on the input tomogram can be visualized in Napari by running `python see_segmentations.py <path_to_segmentation>`. See [Visualizing segmentations from S1](run_picket.md#vis_seg_s1) for detailed instructions on this.

<br/>

---
<br/>

[Back to Home](../README.md)  
[Go to outputs](outputs.md)  
[Go to usage instructions](usage_instructions.md#usage-instructions)  
