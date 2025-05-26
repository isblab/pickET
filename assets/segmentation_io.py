import h5py
import numpy as np


class Segmentations:
    def __init__(self):
        self.semantic_segmentation = None
        self.instance_segmentation = None
        self.metadata = {
            # S1 metadata
            "dataset_name": None,
            "tomogram_path": None,
            "tomogram_shape": None,
            "voxel_size": None,
            "neighborhood_size": None,
            "z_lb_for_clusterer_fitting": None,
            "z_ub_for_clusterer_fitting": None,
            "max_num_neighborhoods_for_fitting": None,
            "fex_mode": None,
            "fex_n_fft_subsets": None,
            "fex_num_neighborhoods_subsets": None,
            "fex_num_sinusoids": None,
            "fex_num_parallel_filters": None,
            "fex_num_output_features": None,
            "clustering_method": None,
            "time_taken_for_fex": None,
            "time_taken_for_clustering": None,
            "time_taken_for_s1": None,
            "timestamp_s1": None,
            # S2 metadata
            "z_lb_for_particle_extraction": None,
            "z_ub_for_particle_extraction": None,
            "particle_cluster_id": None,
            "pex_mode": None,
            "pex_min_distance": None,
            "time_taken_for_s2": None,
            "timestamp_s2": None,
        }

    def generate_output_file(self, out_fname):
        with h5py.File(out_fname, "w") as outf:
            seg_group = outf.create_group("segmentations")
            seg_group.create_dataset(
                "semantic_segmentation", data=self.semantic_segmentation
            )
            if self.instance_segmentation is not None:
                seg_group.create_dataset(
                    "instance_segmentation", data=self.instance_segmentation
                )

            for k, v in self.metadata.items():
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                elif isinstance(v, np.generic):
                    v = v.item()
                elif v == None:
                    v = "None"
                seg_group.attrs[k] = v

    def load_segmentations(self, fname):
        with h5py.File(fname, "r") as infile:
            seg_group = infile["segmentations"]
            for k, v in seg_group.attrs.items():
                self.metadata[k] = v

            self.semantic_segmentation = seg_group[  # type:ignore
                "semantic_segmentation"
            ][:]
            if "instance_segmentation" in seg_group.keys():  # type:ignore
                self.instance_segmentation = seg_group[  # type:ignore
                    "instance_segmentation"
                ]
