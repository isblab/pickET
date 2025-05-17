

# Obtaining particle cluster ID

Look at the each of the six segmentations generated from S1 for the tomograms in the dataset following the workflow described in [visualizing segmentations](visualizing_segmentations.md).  This could be 0 or 1 and is called the `particle_cluster_id` for each of the chosen segmentations. Either of the following ways can be used to obtain the `particle_cluster_id`:

1. Hover over a target particle in the loaded Napari window. This number should appear at the bottom of the window next to the coordinates of the mouse pointer.

2. Check if a target particle is colored with a non-gray color in this overlay. If the particle is colored, then `particle_cluster_id = 1`, else `particle_cluster_id = 0`.

*Note that the `particle_cluster_id` may not be the same for all the segmentations generated from S1 for a given tomogram.*

#TODO: Add Napari tutorial images