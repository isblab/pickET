import sys
import os
import shutil

from modules.config import load_config

from modules.metadata import (
build_dataset
)

from modules.template import (
get_template_voxel_size,
generate_template,
generate_mask
)

from modules.particle_diameter import (
get_particle_diameter
)

from modules.matching import (
build_tm_command,
run_tm_command
)

from modules.extraction import (
build_extraction_command,
run_extraction_command
)

from modules.roc import (
build_roc_command,
run_roc_command
)

from modules.conversion import (
run_conversion
)

from modules.evaluation import (
run_evaluation,
get_threshold_angstrom
)

from modules.benchmark import (
build_benchmark_dataframe,
compute_summary_statistics,
generate_violin_plots,
generate_boxplots
)

def diameter_to_voxels(
    diameter_angstrom,
    voxel_size
):

    return (
        diameter_angstrom /
        voxel_size
    )


def compute_box_size(
    diameter_angstrom,
    voxel_size
):

    diameter_voxels = (
        diameter_to_voxels(
            diameter_angstrom,
            voxel_size
        )
    )

    box_size = int(
        diameter_voxels * 3
    )

    if box_size % 2 != 0:
        box_size += 1

    return box_size


def compute_mask_radius(
    diameter_angstrom,
    voxel_size
):

    diameter_voxels = (
        diameter_to_voxels(
            diameter_angstrom,
            voxel_size
        )
    )

    particle_radius = round(
        diameter_voxels / 2
    )

    mask_radius = int(
        particle_radius * 1.10
    )

    return mask_radius

def main():

  if len(sys.argv) != 2:

    print(
        "Usage: python run_pipeline.py config.yaml"
    )

    sys.exit(1)

config_file = sys.argv[1]

config = load_config(config_file)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

print("\n=== CONFIG LOADED ===\n")

print(
    f"Experiment: "
    f"{config['experiment']['name']}"
)

print(
    f"Dataset: "
    f"{config['dataset']['path']}"
)

print(
    f"Type: "
    f"{config['dataset']['type']}"
)

print(
    f"Template: "
    f"{config['template']['input']}"
)

print(
    f"TM diameter: "
    f"{config['particle']['template_matching_diameter_angstrom']}"
)

print(
    f"Extraction diameter: "
    f"{config['particle']['extraction_diameter_angstrom']}"
)

print(
    f"Extraction cutoff: "
    f"{config['extraction']['cutoff']}"
)
#TODO add args for PDB
print(
    "Low pass: 40 A"
)

print(
    f"GPUs: "
    f"{config['compute']['gpu_ids']}"
)

# --------------------------------------------------
# RESULTS DIRECTORY
# --------------------------------------------------

experiment_dir = (
    config["experiment"]["name"]
)

os.mkdir(
    experiment_dir,
    exist_ok=True
) #TODO crosscheck 

print(
    f"\nExperiment directory: "
    f"{results_dir}"
)

template_dir = os.path.join(
    experiment_dir,
    "template"
)

os.mkdir(
    template_dir,
    exist_ok=True
)

# --------------------------------------------------
# DATASET DISCOVERY
# --------------------------------------------------

dataset_path = config["dataset"]["path"]

dataset = get_metadata(
    dataset_path,
    config
)

tm_particle_diameter = (
    config["particle"]
          ["template_matching_diameter_angstrom"]
)

pdb_file = (
    config["particle"]
          ["pdb"]
)

if tm_particle_diameter is not None:

    print(
        "\nUsing user-provided "
        "particle diameter."
    )

elif pdb_file is not None:

    print(
        "\nEstimating particle "
        "diameter from PDB..."
    )

    tm_particle_diameter = (
        get_particle_diameter(
            pdb_file
        )
    )

    print(
        f"Estimated diameter: "
        f"{tm_particle_diameter} A"
    )

else:

    raise ValueError(

        "\nEither "
        "template_matching_diameter_angstrom "
        "or pdb must be provided."

    )

extraction_particle_diameter = (
    config["particle"]
          ["extraction_diameter_angstrom"]
)

if extraction_particle_diameter == "auto":

    extraction_particle_diameter = (
        tm_particle_diameter
    )

tomogram_voxel_size = (
    dataset[0]["voxel_size"]
)

particle_diameter_voxels = (
    diameter_to_voxels(
        tm_particle_diameter,
        tomogram_voxel_size
    )
)

box_size = compute_box_size(
    tm_particle_diameter,
    tomogram_voxel_size
)

mask_radius = compute_mask_radius(
    tm_particle_diameter,
    tomogram_voxel_size
)

print(
    f"\nTM Particle diameter: "
    f"{tm_particle_diameter:.1f} Å"
)

print(
    f"Extraction diameter: "
    f"{extraction_particle_diameter:.1f} Å"
)

print(
    f"Particle diameter: "
    f"{particle_diameter_voxels:.2f} voxels"
)

print(
    f"Computed box size: "
    f"{box_size}"
)

print(
    f"Computed mask radius: "
    f"{mask_radius}"
)

print("\n=== DATASET DISCOVERY ===\n")

print(
    f"Found {len(dataset)} tomogram(s)\n"
)

for tomo in dataset:

    basename = (
        os.path.splitext(
            os.path.basename(
                tomo["path"]
            )
        )[0]
    )

    tomo_results_dir = os.path.join(
        experiment_dir,
        basename
    )

    os.makedirs(
        tomo_results_dir,
        exist_ok=True
    )

    print(tomo["path"])

    print(
        f"  Shape: "
        f"{tomo['shape']}"
    )

    print(
        f"  Voxel Size: "
        f"{tomo['voxel_size']:.2f} Å"
    )

    print(
        f"  rawtlt: "
        f"{tomo['rawtlt']}"
    )

    print(
        f"  defocus: "
        f"{tomo['defocus']}"
    )

    print(
        f"  dose: "
        f"{tomo['dose']}"
    )

    if tomo["tilt_info"]["mode"] == "rawtlt":

        print(
            f"  Tilt Angles: "
            f"{tomo['tilt_info']['file']}"
        )

    else:

        print(
            f"  Tilt Range: "
            f"{tomo['tilt_info']['min']} "
            f"to "
            f"{tomo['tilt_info']['max']}"
        )

    print()

# --------------------------------------------------
# TEMPLATE GENERATION
# --------------------------------------------------

template_output = os.path.join(
    template_dir,
    config["template_generation"]["output_name"]
)

if config["execution"]["generate_template"]:

    print(
        "\n=== TEMPLATE GENERATION ===\n"
    )

    template_input = (
        config["template"]["input"]
    )

    template_voxel_size = (
        get_template_voxel_size(
            template_input
        )
    )

    tomogram_voxel_size = (
        dataset[0]["voxel_size"]
    )

    print(
        f"Template voxel size: "
        f"{template_voxel_size:.2f} Å"
    )

    print(
        f"Target voxel size: "
        f"{tomogram_voxel_size:.2f} Å"
    )

    generate_template(

        template_input,

        template_output,

        template_voxel_size,

        tomogram_voxel_size,

        box_size,

        config["template_generation"][
            "invert"
        ]
    )

    print(
        "\nTemplate generated:"
    )

    print(template_output)

else:

    print(
        "\nSkipping template generation."
    )

    print(
        f"Using existing template: "
        f"{template_output}"
    )

# --------------------------------------------------
# MASK GENERATION
# --------------------------------------------------

mask_output = os.path.join(
    template_dir,
    config["template_mask"]["output_name"]
)

if config["execution"]["generate_mask"]:

    print(
        "\n=== MASK GENERATION ===\n"
    )

    generate_mask(

        box_size,

        mask_radius,

        mask_output

    )

    print(
        "\nMask generated:"
    )

    print(mask_output)

else:

    print(
        "\nSkipping mask generation."
    )

    print(
        f"Using existing mask: "
        f"{mask_output}"
    )

# --------------------------------------------------
# TEMPLATE MATCHING
# --------------------------------------------------

print(
    "\n=== TEMPLATE MATCHING ===\n"
)

for tomo in dataset:

    basename = (
        os.path.splitext(
            os.path.basename(
                tomo["path"]
            )
        )[0]
    )

    tomo_results_dir = os.path.join(
        experiment_dir,
        basename
    )

    cmd = build_tm_command(

        tomo,

        template_output,

        mask_output,

        tm_particle_diameter,

        config,

        tomo_results_dir

    )

    print(
        "\nGenerated TM command:\n"
    )

    print(
        " ".join(cmd)
    )

    print()

    if config["execution"]["run_template_matching"]:
        run_tm_command(cmd)

    else:
        print("TM execution disabled")


print(
    "\nTM command generation successful."
)

# --------------------------------------------------
# EXTRACTION
# --------------------------------------------------

print(
    "\n=== EXTRACTION ===\n"
)

for tomo in dataset:

    basename = (
        os.path.splitext(
            os.path.basename(
                tomo["path"]
            )
        )[0]
    )

    tomo_results_dir = os.path.join(
        experiment_dir,
        basename
    )

    job_file = os.path.join(
        tomo_results_dir,
        f"{basename}_job.json"
    )

    baseline_cmd = build_extraction_command(

        job_file,

        config,

        extraction_particle_diameter,

        ignore_tomogram_mask = True

    )

    picket_cmd = build_extraction_command(

        job_file,

        config,

        extraction_particle_diameter,

        ignore_tomogram_mask = False

    )


    print(
        "\nGenerated Baseline Extraction Command:\n"
    )

    print(
        " ".join(baseline_cmd)
    )

    print(
        "\nGenerated PickET Extraction Command:\n"
    )

    print(
        " ".join(picket_cmd)
    )

    print()

    if config["execution"][
        "run_extraction"
    ]:

        run_extraction_command(baseline_cmd)


        baseline_star = os.path.join(
            tomo_results_dir,
            f"{basename}_particles.star"
        )

        baseline_graph = os.path.join(
            tomo_results_dir,
            f"{basename}_extraction_graph.svg"
        )

        shutil.move(

            baseline_star,

            os.path.join(
                tomo_results_dir,
                "baseline_particles.star"
            )

        )

        shutil.move(

            baseline_graph,

            os.path.join(
                tomo_results_dir,
                "baseline_extraction_graph.svg"
            )

        )

        run_extraction_command(
        picket_cmd
        )

        picket_star = os.path.join(
            tomo_results_dir,
            f"{basename}_particles.star"
        )

        picket_graph = os.path.join(
            tomo_results_dir,
            f"{basename}_extraction_graph.svg"
        )

        shutil.move(
            picket_star,
            os.path.join(
                tomo_results_dir,
                "picket_particles.star"
            )
        )

        shutil.move(
            picket_graph,
            os.path.join(
                tomo_results_dir,
                "picket_extraction_graph.svg"
            )
        )

    else:
       print(
           "Extraction disabled."
       )

# --------------------------------------------------
# ROC
# --------------------------------------------------

print(
    "\n=== ROC ===\n"
)

for tomo in dataset:

    basename = (
        os.path.splitext(
            os.path.basename(
                tomo["path"]
            )
        )[0]
    )

    tomo_results_dir = os.path.join(
        experiment_dir,
        basename
    )

    job_file = os.path.join(
        tomo_results_dir,
        f"{basename}_job.json"
    )

    baseline_cmd = build_roc_command(

        job_file,

        config,

        ignore_tomogram_mask=True

    )

    picket_cmd = build_roc_command(

        job_file,

        config,

        ignore_tomogram_mask=False

    )

    print(
        "\nGenerated Baseline ROC Command:\n"
    )

    print(
        " ".join(baseline_cmd)
    )

    print(
        "\nGenerated PickET ROC Command:\n"
    )

    print(
        " ".join(picket_cmd)
    )

    print()

    if config["execution"][
        "run_roc"
    ]:

        baseline_log = os.path.join(
            tomo_results_dir,
            "roc.log"
        )

        run_roc_command(
            baseline_cmd,
            baseline_log
        )

        shutil.move(

            os.path.join(
                tomo_results_dir,
                "roc.log"
            ),

            os.path.join(
                tomo_results_dir,
                "baseline_roc.log"
            )

        )

        shutil.move(

            os.path.join(
                tomo_results_dir,
                f"{basename}_roc.svg"
            ),

            os.path.join(
                tomo_results_dir,
                "baseline_roc.svg"
            )

        )

        picket_log = os.path.join(
            tomo_results_dir,
            "roc.log"
        )

        run_roc_command(
            picket_cmd,
            picket_log
        )

        shutil.move(

            os.path.join(
                tomo_results_dir,
                "roc.log"
            ),

            os.path.join(
                tomo_results_dir,
                "picket_roc.log"
            )

        )

        shutil.move(

            os.path.join(
                tomo_results_dir,
                f"{basename}_roc.svg"
            ),

            os.path.join(
                tomo_results_dir,
                "picket_roc.svg"
            )

        )

    else:

        print(
            "ROC disabled."
        )

# --------------------------------------------------
# CONVERSION
# --------------------------------------------------

print(
    "\n=== CONVERSION ===\n"
)

for tomo in dataset:

    basename = (
        os.path.splitext(
            os.path.basename(
                tomo["path"]
            )
        )[0]
    )

    tomo_results_dir = os.path.join(
        experiment_dir,
        basename
    )

    baseline_star = os.path.join(

        tomo_results_dir,

        "baseline_particles.star"

    )

    baseline_yaml = os.path.join(

        tomo_results_dir,

        "baseline_prediction.yaml"

    )

    picket_star = os.path.join(

        tomo_results_dir,

        "picket_particles.star"

    )

    picket_yaml = os.path.join(

        tomo_results_dir,

        "picket_prediction.yaml"

    )

    print(
        f"\nBaseline STAR: {baseline_star}"
    )

    print(
        f"Baseline YAML: {baseline_yaml}"
    )

    print(
        f"\nPickET STAR: {picket_star}"
    )

    print(
        f"PickET YAML: {picket_yaml}"
    )

    print()

    if config["execution"][
        "run_conversion"
    ]:

        run_conversion(

            baseline_star,

            baseline_yaml,

            config["conversion"][
                "particle_name"
            ],

            tomo["shape"],

            tomo["path"],

            tomo["voxel_size"]

        )

        run_conversion(

            picket_star,

            picket_yaml,

            config["conversion"][
                "particle_name"
            ],

            tomo["shape"],

            tomo["path"],

            tomo["voxel_size"]

        )


    else:

        print(
            "Conversion disabled."
        )

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

print(
    "\n=== EVALUATION ===\n"
)

for tomo in dataset:

    basename = (

        os.path.splitext(

            os.path.basename(

                tomo["path"]

            )

        )[0]

    )

    ground_truth_ndjson = os.path.join(
        config["ground_truth"][
            "directory"
        ],

        f"{basename}.ndjson"

    )

    if not os.path.exists(
        ground_truth_ndjson
    ):

        raise FileNotFoundError(

            f"Missing GT file:\n"
            f"{ground_truth_ndjson}"
        )

    tomo_results_dir = os.path.join(
        experiment_dir,
        basename
    )

    baseline_prediction_yaml = os.path.join(

        tomo_results_dir,

        "baseline_prediction.yaml"

    )

    baseline_evaluation_yaml = os.path.join(

        tomo_results_dir,

        "baseline_evaluation.yaml"

    )

    picket_prediction_yaml = os.path.join(

        tomo_results_dir,

        "picket_prediction.yaml"

    )

    picket_evaluation_yaml = os.path.join(

        tomo_results_dir,

        "picket_evaluation.yaml"

    )

    print(
        f"Ground Truth: "
        f"{ground_truth_ndjson}"
    )

    print(
        f"\nBaseline Prediction: "
        f"{baseline_prediction_yaml}"
    )

    print(
        f"Baseline Output: "
        f"{baseline_evaluation_yaml}"
    )

    print(
        f"\nPickET Prediction: "
        f"{picket_prediction_yaml}"
    )

    print(
        f"PickET Output: "
        f"{picket_evaluation_yaml}"
    )

    threshold_angstrom = (
        get_threshold_angstrom(
            config[
                "dataset"
            ][
                 "type"
            ]
        )
    )

    print(
        f"\nUsing threshold: "
        f"{threshold_angstrom} A"
    )

    if config["execution"][
        "run_evaluation"
    ]:

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

        print(
            "Evaluation disabled."
        )

# --------------------------------------------------
# BENCHMARK SUMMARY
# --------------------------------------------------

print(
    "\n=== BENCHMARK SUMMARY ===\n"
)

if config["execution"][
    "run_benchmark_summary"
]:

    df = build_benchmark_dataframe(

        dataset,

        experiment_dir

    )

    print(df)

    csv_file = os.path.join(

        experiment_dir,

        "benchmark_summary.csv"

    )

    df.to_csv(

        csv_file,

        index=False

    )

    print(
        f"\nSaved: {csv_file}"
    )

    summary_df = (
        compute_summary_statistics(
            df
        )
    )

    summary_file = os.path.join(

        experiment_dir,

        "benchmark_statistics.csv"

    )

    summary_df.to_csv(

        summary_file,

        index=False

    )

    print(
        f"Saved: {summary_file}"
    )

    generate_violin_plots(

        df,

        experiment_dir

    )

    generate_boxplots(

        df,

        experiment_dir

    )

    print(
        "\nPlots generated."
    )

else:

    print(
        "Benchmark summary disabled."
    )

if __name__ == "__main__":
    main()
