# Visualizing segmentations

Now, run the following command on each of the segmentations to visualize an overlay of the segmentation on the input tomogram in Napari:  
```bash
python src/picket/accessories/see_segmentations.py <path_to_segmentation> <segmentation_type>
```

where `segmentation_type` is either `semantic_segmentation` or `instance_segmentation`.

!!! note "Note"
    *This step displays the segmentation overlayed on the input tomogram in a [Napari](https://napari.org/) window.*  

<div style="display: flex; justify-content: center;">
    <div align="center" style="margin-right: 100px;">
        <img src="images/semantic_segmentation.png" alt="Fig. 2A: Output from S1 - Semantic segmentation" height="300" align="center">
        <p align="center"><span class="caption">Fig. 2A: Output from S1 - Semantic segmentation</span></p>
    </div>
    <div align="center">
        <img src="images/instance_segmentation.png" alt="Fig. 2B: Output from S2 - Instance segmentation" height="300" align="center">
        <p align="center"><span class="caption">Fig. 2B: Output from S2 - Instance segmentation</span></p>
    </div>
</div>
<br/>

---

[Back to Home](index.md)  
[Go to usage instructions](usage_instructions.md)  
[Go to obtaining particle cluster ID](obtaining_particle_cluster_id.md)  