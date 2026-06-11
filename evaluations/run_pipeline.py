import sys
import os
import shutil

from modules.config import load_config

from modules.metadata import get_metadata

from modules.template import (
    get_template_voxel_size,
    generate_template,
    generate_mask
)

from modules.particle_diameter import get_particle_diameter

from modules.matching import (
    build_tm_command,
    run_tm_command
)

from modules.extraction import (
    build_extraction_command,
    run_extraction_command
)

from modules.roc import (
    count_particles,
    build_roc_command,
    run_roc_command
)

from modules.evaluation import (
    run_conversion,
    run_evaluation,
    get_threshold_angstrom
)

from modules.benchmark import (
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


def rename_roc_outputs(
    tomo_results_dir,
    basename,
    prefix,
):

    shutil.move(
         os.path.join(
              tomo_results_dir, "roc.log"
         ),
         os.path.join(
              tomo_results_dir, f"{prefix}_roc.log"
         ),
    )

    shutil.move(
         os.path.join(
              tomo_results_dir, f"{basename}_roc.svg"
         ),
         os.path.join(
              tomo_results_dir, f"{prefix}_roc.svg"
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
    os.mkdir(experiment_dir)
    template_dir = os.path.join(experiment_dir, "template")
    os.mkdir(template_dir)

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

    if extraction_particle_diameter is None:
        extraction_particle_diameter = tm_particle_diameter

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
        os.mkdir(tomo_results_dir)

    # --------------------------------------------------
    # TEMPLATE GENERATION
    # --------------------------------------------------

    template_output = os.path.join(
        template_dir,
        config["template_generation"]["output_name"]
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
        config["template_mask"]["output_name"]
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
            ignore_tomogram_mask=True
        )

        picket_cmd = build_extraction_command(
            job_file,
            config,
            extraction_particle_diameter,
            ignore_tomogram_mask=False
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
    # ROC
    # --------------------------------------------------

    print("\n=== ROC ===\n")

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
        baseline_star = os.path.join(tomo_results_dir, "baseline_particles.star")
        baseline_n = 3*(count_particles(baseline_star))
        picket_star = os.path.join(tomo_results_dir, "picket_particles.star")
        picket_n = 3*(count_particles(picket_star))

        baseline_cmd = build_roc_command(
            job_file,
            config,
            extraction_particle_diameter,
            baseline_n,
            ignore_tomogram_mask=True
        )

        picket_cmd = build_roc_command(
            job_file,
            config,
            extraction_particle_diameter,
            picket_n,
            ignore_tomogram_mask=False
        )

        print(f"Baseline ROC particles: {baseline_n}")
        print(f"PickET ROC particles: {picket_n}")
        print(f"\nGenerated Baseline ROC Command:")
        print(" ".join(baseline_cmd))
        print(f"\nGenerated PickET ROC Command:")
        print(" ".join(picket_cmd))

        if config["execution"]["run_roc"]:
            baseline_log = os.path.join(tomo_results_dir, "roc.log")
            run_roc_command(baseline_cmd, baseline_log)
            rename_roc_outputs(tomo_results_dir, basename, "baseline")

            picket_log = os.path.join(tomo_results_dir, "roc.log")
            run_roc_command(picket_cmd, picket_log)
            rename_roc_outputs(tomo_results_dir, basename, "picket")

        else:
            print("ROC disabled.")

    # --------------------------------------------------
    # STAR TO YAML PREDICTION FILE CONVERSION
    # --------------------------------------------------

    print("\n=== STAR TO YAML PREDICTION FILE CONVERSION ===\n")

    for tomo in dataset:

        basename = (
            os.path.splitext(
                os.path.basename(
                    tomo["path"]
                )
            )[0]
        )

        tomo_results_dir = os.path.join(experiment_dir, basename)
        baseline_star = os.path.join(
            tomo_results_dir, "baseline_particles.star")
        baseline_yaml = os.path.join(
            tomo_results_dir, "baseline_prediction.yaml")
        picket_star = os.path.join(
            tomo_results_dir, "picket_particles.star")
        picket_yaml = os.path.join(
            tomo_results_dir, "picket_prediction.yaml")

        if config["execution"]["run_conversion"]:
            run_conversion(
                baseline_star,
                baseline_yaml,
                config["conversion"]["particle_name"],
                tomo["shape"],
                tomo["path"],
                tomo["voxel_size"]
            )

            run_conversion(
                picket_star,
                picket_yaml,
                config["conversion"]["particle_name"],
                tomo["shape"],
                tomo["path"],
                tomo["voxel_size"]
            )

        else:
            print("Conversion disabled.")

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
        baseline_prediction_yaml = os.path.join(
            tomo_results_dir, "baseline_prediction.yaml")
        baseline_evaluation_yaml = os.path.join(
            tomo_results_dir, "baseline_evaluation.yaml")
        picket_prediction_yaml = os.path.join(
            tomo_results_dir, "picket_prediction.yaml")
        picket_evaluation_yaml = os.path.join(
            tomo_results_dir, "picket_evaluation.yaml")

        threshold_angstrom = (get_threshold_angstrom(
            config["dataset"]["type"]))
        print(f"\nUsing threshold: {threshold_angstrom} A")

        if config["execution"]["run_evaluation"]:
            run_evaluation(
                baseline_prediction_yaml,
                ground_truth_ndjson,
                threshold_angstrom,
                baseline_evaluation_yaml
            )

            run_evaluation(
                picket_prediction_yaml,
                ground_truth_ndjson,
                threshold_angstrom,
                picket_evaluation_yaml
            )

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
