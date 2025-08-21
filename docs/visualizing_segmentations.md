# Visualizing segmentations

Now, run the following command on each of the segmentations to visualize an overlay of the segmentation on the input tomogram in Napari:  
```bash
python src/picket/accessories/see_segmentations.py <path_to_segmentation> <segmentation_type>
```

where `segmentation_type` is either `semantic_segmentation` or `instance_segmentation`.

!!! note "Note"
    *This step displays the segmentation overlayed on the input tomogram in a [Napari](https://napari.org/) window.*  

![Fig. 3A: Output from S1 - Semantic segmentation](images/semantic_segmentation.png){: style="width: 300px; display: block; margin-left: auto; margin-right: auto;"}
<div align="center">
    <p align="center"><span class="caption">Fig. 2A: Output from S1 - Semantic segmentation</span></p>
</div>

![Fig. 3B: Output from S2 - Instance segmentation](images/instance_segmentation.png){: style="width: 300px; display: block; margin-left: auto; margin-right: auto;"}
<div align="center">
    <p align="center"><span class="caption">Fig. 2B: Output from S2 - Instance segmentation</span></p>
</div>
<br/>

---

[Back to Home](index.md)  
[Go to usage instructions](usage_instructions.md)  
[Go to obtaining particle cluster ID](obtaining_particle_cluster_id.md)  