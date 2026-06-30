import sys
import os
import shutil

from TM_modules.config import load_config

from TM_modules.metadata import get_metadata

from TM_modules.preprocessing import run_preprocessing

from TM_modules.annotation_conversion import convert_annotations

from TM_modules.template import (
    get_template_voxel_size,
    get_particle_diameter,
    generate_template,
    generate_mask
)

from TM_modules.matching import (
    build_tm_command,
    run_tm_command
)

from TM_modules.extraction import (
    get_extraction_diameter,
    build_extraction_command,
    run_extraction_command
)

from TM_modules.evaluation import (
    run_evaluation,
    get_threshold_angstrom
)

from TM_modules.benchmark import (
    build_benchmark_dataframe,
    compute_summary_statistics,
    generate_violin_plots,
    generate_boxplots
)


def rename_extraction_outputs(
    tomo_results_dir,
    basename,
    prefix,
):

    shutil.move(
         os.path.join(
              tomo_results_dir,
              f"{basename}_particles.star"
         ),
         os.path.join(
              tomo_results_dir,
              f"{prefix}_particles.star"
         ),
    )

    shutil.move(
         os.path.join(
              tomo_results_dir,
              f"{basename}_extraction_graph.svg"
         ),
         os.path.join(
              tomo_results_dir,
              f"{prefix}_extraction_graph.svg"
         ),
    )


def main():

    if len(sys.argv) != 2:
        print(
            "Usage: python run_pipeline.py config.yaml"
        )

        sys.exit(1)

    config_file = sys.argv[1]
    config = load_config(config_file)

    # --------------------------------------------------
    # RESULTS DIRECTORY
    # --------------------------------------------------

    experiment_dir = config["experiment"]["name"]
    os.makedirs(experiment_dir, exist_ok=True)
    template_dir = os.path.join(experiment_dir, "template")
    os.makedirs(template_dir, exist_ok=True)

    # --------------------------------------------------
    #  PREPROCESSING
    # --------------------------------------------------

    if config["execution"]["run_preprocessing"]:
        print("\n=== PREPROCESSING ===\n")

        run_preprocessing(
            tomogram_folder=config["dataset"]["path"],
            picket_in_h5=
                config["preprocessing"]["picket_in_h5"],
            picket_out_mrc=
                config["preprocessing"]["picket_out_mrc"],
                tomogram_config=config.get("tomograms", {})
        )

    else:
        print("\nPreprocessing disabled.")

    # --------------------------------------------------
    # ANNOTATION CONVERSION
    # --------------------------------------------------

    if config["execution"]["run_annotation_conversion"]:
        print("\n=== ANNOTATION CONVERSION ===\n")

        convert_annotations(
            annotation_folder =
                config["gt_annotation_conversion"][
                    "annotation_folder"
                ],
            output_folder=
                config["ground_truth"]["directory"],
            annotation_suffix=
                config["gt_annotation_conversion"][
                    "annotation_suffix"
                ]
        )

    else:
        print("\nAnnotation conversion disabled.")

    # --------------------------------------------------
    # DATASET DISCOVERY
    # --------------------------------------------------

    dataset_path = config["dataset"]["path"]
    dataset = get_metadata(dataset_path, config)
    tm_particle_diameter = config["particle"][
        "template_matching_diameter_angstrom"]
    pdb_file = config["particle"]["pdb"]

    if tm_particle_diameter is not None:
        print("\nUsing user-provided particle diameter.")

    elif pdb_file is not None:
        print("\nEstimating particle diameter from PDB...")
        tm_particle_diameter = get_particle_diameter(pdb_file)
        print(f"Estimated diameter: {tm_particle_diameter} A")

    else:
        raise ValueError(
            "Either template_matching_diameter_angstrom "
            "or pdb must be provided."
        )

    extraction_particle_diameter = config[
        "particle"
    ]["extraction_diameter_angstrom"]

    if extraction_particle_diameter is not None:
        print("\nUsing user-provided extraction diameter.")

    elif config["particle"][
        "extraction_diameter_required"
    ]:
        if pdb_file is None:
            raise ValueError(
                "PDB file required for "
                "RG-based extraction diameter."
            )

        extraction_particle_diameter = get_extraction_diameter(pdb_file)
        print("\nUsing RG-based extraction diameter.")

    else:
        extraction_particle_diameter = tm_particle_diameter
        print("\nUsing template matching diameter for extraction.")

    print(f"Extraction diameter: {extraction_particle_diameter} A")

    tomogram_voxel_size = dataset[0]["voxel_size"]
    particle_diameter_voxels = tm_particle_diameter/tomogram_voxel_size
    box_size = int(particle_diameter_voxels*3)
    if box_size % 2 != 0:
        box_size += 1
    particle_radius = round(particle_diameter_voxels/2)
    mask_radius = int(particle_radius*1.1)

    print(f"Particle diameter: {particle_diameter_voxels:.2f} voxels")
    print(f"Computed box size: {box_size}")
    print(f"Computed mask radius: {mask_radius}")

    print("\n=== DATASET DISCOVERY ===\n")
    print(f"Found {len(dataset)} tomogram(s)\n")

    for tomo in dataset:

        basename = (
            os.path.splitext(
                os.path.basename(
                    tomo["path"]
                )
            )[0]
        )

        tomo_results_dir = os.path.join(experiment_dir, basename)
        os.makedirs(tomo_results_dir, exist_ok=True)

    # --------------------------------------------------
    # TEMPLATE GENERATION
    # --------------------------------------------------

    template_output = os.path.join(
        template_dir,
        config["template_generation"]["output_template_file"]
    )

    if config["execution"]["generate_template"]:
        print("\n=== TEMPLATE GENERATION ===\n")

        template_input = config["template"]["input"]
        template_voxel_size = get_template_voxel_size(template_input)
        tomogram_voxel_size = dataset[0]["voxel_size"]

        generate_template(
            template_input,
            template_output,
            template_voxel_size,
            tomogram_voxel_size,
            box_size,
            config["template_generation"]["invert"]
        )

        print(f"\nTemplate generated: {template_output}")

    else:
        print("\nSkipping template generation.")
        print(f"Using existing template: {template_output}")

    # --------------------------------------------------
    # MASK GENERATION
    # --------------------------------------------------

    mask_output = os.path.join(
        template_dir,
        config["template_generation"]["output_mask_file"]
    )

    if config["execution"]["generate_mask"]:
        print("\n=== MASK GENERATION ===\n")

        generate_mask(
            box_size,
            mask_radius,
            mask_output
        )

        print(f"\nMask generated: {mask_output}")

    else:
        print(f"\nSkipping mask generation.")
        print(f"Using existing mask: {mask_output}")

    # --------------------------------------------------
    # TEMPLATE MATCHING
    # --------------------------------------------------

    print("\n=== TEMPLATE MATCHING ===\n")

    for tomo in dataset:

        basename = (
            os.path.splitext(
                os.path.basename(
                    tomo["path"]
                )
            )[0]
        )

        tomo_results_dir = os.path.join(experiment_dir, basename)

        cmd = build_tm_command(
            tomo,
            template_output,
            mask_output,
            tm_particle_diameter,
            config,
            tomo_results_dir
        )

        print("\nGenerated TM command:")
        print(" ".join(cmd))

        if config["execution"]["run_template_matching"]:
            run_tm_command(cmd)

        else:
            print("TM execution disabled")

    # --------------------------------------------------
    # EXTRACTION
    # --------------------------------------------------

    print("\n=== EXTRACTION ===\n")

    for tomo in dataset:

        basename = (
            os.path.splitext(
                os.path.basename(
                    tomo["path"]
                )
            )[0]
        )

        tomo_results_dir = os.path.join(experiment_dir, basename)
        job_file = os.path.join(tomo_results_dir, f"{basename}_job.json")

        baseline_cmd = build_extraction_command(
            job_file,
            config,
            extraction_particle_diameter,
            tomogram_mask=None
        )

        picket_cmd = build_extraction_command(
            job_file,
            config,
            extraction_particle_diameter,
            tomogram_mask=tomo["mask_path"]
        )

        print("\nGenerated Baseline Extraction Command:")
        print(" ".join(baseline_cmd))
        print("\nGenerated PickET Extraction Command:")
        print(" ".join(picket_cmd))

        if config["execution"]["run_extraction"]:
            run_extraction_command(baseline_cmd)
            rename_extraction_outputs(tomo_results_dir, basename, "baseline")
            run_extraction_command(picket_cmd)
            rename_extraction_outputs(tomo_results_dir, basename, "picket")

        else:
            print("Extraction disabled.")

    # --------------------------------------------------
    # EVALUATION
    # --------------------------------------------------

    print("\n=== EVALUATION ===\n")

    for tomo in dataset:

        basename = (
            os.path.splitext(
                os.path.basename(
                    tomo["path"]
                )
            )[0]
        )

        ground_truth_ndjson = os.path.join(
            config["ground_truth"]["directory"],
            f"{basename}.ndjson"
        )

        if not os.path.exists(ground_truth_ndjson):
            raise FileNotFoundError(
                f"Missing GT file:\n {ground_truth_ndjson}"
            )

        tomo_results_dir = os.path.join(experiment_dir, basename)
        baseline_star = os.path.join(tomo_results_dir, "baseline_particles.star")
        baseline_evaluation_yaml = os.path.join(tomo_results_dir, "baseline_evaluation.yaml")
        picket_star = os.path.join(tomo_results_dir, "picket_particles.star")
        picket_evaluation_yaml = os.path.join(tomo_results_dir, "picket_evaluation.yaml")

        threshold_angstrom = (get_threshold_angstrom(
            config["dataset"]["type"]))
        print(f"\nUsing threshold: {threshold_angstrom} A")

        if config["execution"]["run_evaluation"]:
            if os.path.exists(baseline_star):
                run_evaluation(
                    baseline_star,
                    ground_truth_ndjson,
                    threshold_angstrom,
                    baseline_evaluation_yaml,
                    tomo["shape"],
                    tomo["voxel_size"]
                )
            else:
                print("Skipping baseline evaluation.")

            if os.path.exists(picket_star):
                run_evaluation(
                    picket_star,
                    ground_truth_ndjson,
                    threshold_angstrom,
                    picket_evaluation_yaml,
                    tomo["shape"],
                    tomo["voxel_size"]
                )
            else:
                print("Skipping picket evaluation.")

        else:
            print("Evaluation disabled.")

    # --------------------------------------------------
    # BENCHMARK SUMMARY
    # --------------------------------------------------

    print("\n=== BENCHMARK SUMMARY ===\n")

    if config["execution"]["run_benchmark_summary"]:
        df = build_benchmark_dataframe(dataset, experiment_dir)
        csv_file = os.path.join(experiment_dir, "benchmark_summary.csv")
        df.to_csv(csv_file, index=False)
        summary_df = (compute_summary_statistics(df))
        summary_file = os.path.join(experiment_dir, "benchmark_statistics.csv")
        summary_df.to_csv(summary_file, index=False)
        generate_violin_plots(df, experiment_dir)
        generate_boxplots(df, experiment_dir)

    else:
        print("Benchmark summary disabled.")

if __name__ == "__main__":
    main()
