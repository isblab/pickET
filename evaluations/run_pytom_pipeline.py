import sys
import os
import time
import shutil
import argparse
import warnings

from TM_modules.config import load_config

from TM_modules.metadata import (
    get_metadata,
    get_result_dirname,
)

from TM_modules.preprocessing import (
    run_preprocessing,
    get_tomogram_files,
)

from TM_modules.annotation_conversion import (
    convert_annotations,
    convert_tomotwin_annotation,
)

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

warnings.filterwarnings("ignore")


def rename_extraction_outputs(
    tomo_results_dir,
    basename,
    prefix,
):

    shutil.move(
        os.path.join(tomo_results_dir, f"{basename}_particles.star"),
        os.path.join(tomo_results_dir, f"{prefix}_particles.star"),
    )

    shutil.move(
        os.path.join(tomo_results_dir, f"{basename}_extraction_graph.svg"),
        os.path.join(tomo_results_dir, f"{prefix}_extraction_graph.svg"),
    )


def main(config, outdir=None):

    experiment = config["experiment"]["name"]
    dataset_path = config["dataset"]["path"]
    dataset_type = config["dataset"]["type"]

    if outdir is None:
        experiment_dir = experiment
    else:
        experiment_dir = os.path.join(outdir, experiment)

    template_dir = os.path.join(experiment_dir, "template")

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(template_dir, exist_ok=True)

    # --------------------------------------------------
    # DATASET DISCOVERY
    # --------------------------------------------------

    tomogram_files=get_tomogram_files(dataset_path, dataset_type)
    dataset = get_metadata(tomogram_files, config)
    tm_particle_diameter = get_particle_diameter(config)
    extraction_particle_diameter = get_extraction_diameter(config)

    if extraction_particle_diameter is None:
        extraction_particle_diameter = tm_particle_diameter
        print("\nUsing template matching diameter for extraction.")

    print(f"Extraction diameter: {extraction_particle_diameter} A")

    tomogram_voxel_size = dataset[0]["voxel_size"] #!? assumes all tomograms have equal voxel size
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

    # --------------------------------------------------
    #  PREPROCESSING
    # --------------------------------------------------

    print("\n=== PREPROCESSING ===\n")
    if config["execution"]["run_preprocessing"]:

        run_preprocessing(
            tomogram_files=tomogram_files,
            picket_in_h5=config["preprocessing"]["picket_in_h5"],
            picket_out_mrc=config["preprocessing"]["picket_out_mrc"],
            tomogram_config=config.get("tomograms", {}),
            dataset_type=dataset_type,
        )

    else:
        print("\nPreprocessing disabled.")

    # --------------------------------------------------
    # ANNOTATION CONVERSION
    # --------------------------------------------------

    print("\n=== ANNOTATION CONVERSION ===\n")
    if config["execution"]["run_annotation_conversion"]:

        if config["dataset"]["type"] in ["experimental", "tutorial"]:

            convert_annotations(
                annotation_folder=config["gt_annotation_conversion"]["annotation_folder"],
                output_folder=config["ground_truth"]["directory"],
                annotation_suffix=config["gt_annotation_conversion"]["annotation_suffix"]
            )

        elif config["dataset"]["type"] == "simulated":

            particle_pdb = os.path.splitext(
                os.path.basename(config["particle"]["pdb"])
            )[0]

            convert_tomotwin_annotation(
                tomotwin_sim_round_path=dataset_path,
                output_folder=config["ground_truth"]["directory"],
                particles=[particle_pdb],
                ignore_vesicle=True,
                ignore_fiducial=True,
            )

    else:
        print("\nAnnotation conversion disabled.")

    # --------------------------------------------------
    # TEMPLATE GENERATION
    # --------------------------------------------------

    template_output = os.path.join(
        template_dir,
        config["template_generation"]["output_template_file"]
    )

    print("\n=== TEMPLATE GENERATION ===\n")
    if config["execution"]["generate_template"]:

        template_input = config["template"]["input"]

        generate_template(
            template_inpath=template_input,
            template_outpath=template_output,
            output_voxel_size=tomogram_voxel_size,
            box_size=box_size,
            invert=config["template_generation"]["invert"]
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

    print("\n=== MASK GENERATION ===\n")
    if config["execution"]["generate_mask"]:

        generate_mask(
            box_size=box_size,
            radius=mask_radius,
            output_mask=mask_output,
        )

        print(f"\nMask generated: {mask_output}")

    else:
        print(f"\nSkipping mask generation.")
        print(f"Using existing mask: {mask_output}")

    for tomo in dataset:

        # --------------------------------------------------
        # TEMPLATE MATCHING
        # --------------------------------------------------
        print("\n=== TEMPLATE MATCHING ===\n")

        tomogram_path = tomo["path"]
        result_name = get_result_dirname(dataset_type, tomogram_path)
        tomo_results_dir = os.path.join(experiment_dir, result_name)
        os.makedirs(tomo_results_dir, exist_ok=True)

        cmd = build_tm_command(
            tomo,
            template_output,
            mask_output,
            tm_particle_diameter,
            config,
            tomo_results_dir
        )

        if config["execution"]["run_template_matching"]:
            _tm_start = time.perf_counter()
            print("\nGenerated TM command:")
            print(" ".join(cmd))
            run_tm_command(cmd)
            _tm_end = time.perf_counter()
            with open(os.path.join(tomo_results_dir, "time.log"), "w") as f:
                f.write(str(f"{(_tm_end - _tm_start):.3f}"))

        else:
            print("TM execution disabled")

        # --------------------------------------------------
        # EXTRACTION
        # --------------------------------------------------
        print("\n=== EXTRACTION ===\n")

        basename = os.path.splitext(os.path.basename(tomo["path"]))[0]
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

        if config["execution"]["run_extraction"]:

            print("\nGenerated Baseline Extraction Command:")
            print(" ".join(baseline_cmd))
            print("\nGenerated PickET Extraction Command:")
            print(" ".join(picket_cmd))

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

        ground_truth_ndjson = os.path.join(
            config["ground_truth"]["directory"],
            f"{result_name}.ndjson"
        )

        baseline_star = os.path.join(tomo_results_dir, "baseline_particles.star")
        baseline_evaluation_yaml = os.path.join(tomo_results_dir, "baseline_evaluation.yaml")
        picket_star = os.path.join(tomo_results_dir, "picket_particles.star")
        picket_evaluation_yaml = os.path.join(tomo_results_dir, "picket_evaluation.yaml")
        threshold_angstrom = get_threshold_angstrom(config["dataset"]["type"])

        if config["execution"]["run_evaluation"]:

            print(f"\nUsing threshold: {threshold_angstrom} A")

            if not os.path.exists(ground_truth_ndjson):
                raise FileNotFoundError(
                    f"Missing GT file:\n {ground_truth_ndjson}"
                )

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

    if config["execution"]["run_benchmark_summary"]:
        df = build_benchmark_dataframe(experiment_dir)
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

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to tm_config.yaml"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        required=False,
        help="Directory to save output"
    )

    args = parser.parse_args()

    config = load_config(args.config)

    if args.outdir is not None:
        os.makedirs(args.outdir, exist_ok=True)

    main(config, args.outdir)
