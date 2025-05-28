# Visualizing segmentations from S1: <a name="vis_seg_s1"></a>

***Note:*** *This step launches the [Napari](https://napari.org/) GUI. If the PickET workflow is being run on a remote computing node, we recommend users to connect to the remote computing node using `ssh` with `-X` flag. In other words, connect to the remote computing node using the following command:*  

```
ssh -X <username>@<ip_address of the remote computing node>
```

<br/>

Now, run the following command on each of the segmentations to visualize an overlay of the segmentation on the input tomogram in Napari:  
```
python see_segmentations.py <path_to_segmentation> <segmentation_type>
```
where `segmentation_type` is either `semantic_segmentation` or `instance_segmentation`.

<div style="display: flex; justify-content: center;">
    <div align="center" style="margin-right: 100px;">
        <img src="../images/semantic_segmentation.png" alt="Fig. 2A: Output from S1 - Semantic segmentation" height="300" align="center">
        <p align="center"><b>Fig. 2A: Output from S1 - Semantic segmentation</b></p>
    </div>
    <div align="center">
        <img src="../images/instance_segmentation.png" alt="Fig. 2B: Output from S2 - Instance segmentation" height="300" align="center">
        <p align="center"><b>Fig. 2B: Output from S2 - Instance segmentation</b></p>
    </div>
</div>
<br/>

---
<br/>

[Back to Home](../README.md)  
[Go to usage instructions](usage_instructions.md#usage-instructions)  
[Go to obtaining particle cluster ID](obtaining_particle_cluster_id.md)  