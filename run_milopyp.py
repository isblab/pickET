import yaml
import time
import subprocess


def main():
    input_fname_mapping = {
        # "10001": "input_10001.txt",
        # "10008": "input_10008.txt",
        "10301": "input_10301.txt",
        # "10440": "input_10440.txt",
        # "tomotwin": "input_tomotwin.txt",
    }
    time_taken = {}

    for dataset_id, input_fname in input_fname_mapping.items():
        s1_cmd = [
            "python",
            "/home/shreyas/Projects/mining_tomograms/milopyp/cet_pick/cet_pick/simsiam_main.py",
            "simsiam3d",
            "--num_epochs",
            "20",
            "--exp_id",
            dataset_id,
            "--bbox",
            "36",
            "--dataset",
            "simsiam3d",
            "--arch",
            "simsiam2d_18",
            "--lr",
            "1e-3",
            "--train_img_txt",
            input_fname,
            "--batch_size",
            "256",
            "--val_intervals",
            "20 ",
            "--save_all",
            "--gauss",
            "0.8",
            "--dog",
            "3,5",
            "--order",
            "xyz",
        ]

        tic_s1 = time.perf_counter()
        s1_out = subprocess.run(s1_cmd)
        toc_s1 = time.perf_counter()

        if s1_out.returncode != 0:
            raise RuntimeError(f"S1 failed for {dataset_id}\nTerminating...")

        time_taken[dataset_id] = {"s1": toc_s1 - tic_s1}

        s2_cmd = [
            "python",
            "/home/shreyas/Projects/mining_tomograms/milopyp/cet_pick/cet_pick/simsiam_test_hm_3d.py",
            "simsiam3d",
            "--exp_id",
            dataset_id,
            "--bbox",
            "36",
            "--dataset",
            "simsiam3d",
            "--arch",
            "simsiam2d_18",
            "--test_img_txt",
            input_fname,
            "--load_model",
            f"exp/simsiam3d/{dataset_id}/model_20.pth",
            "--gauss",
            "0.8",
            "--dog",
            "3,5",
            "--order",
            "xyz",
        ]
        tic_s2 = time.perf_counter()
        s2_out = subprocess.run(s2_cmd)
        toc_s2 = time.perf_counter()

        if s2_out.returncode != 0:
            raise RuntimeError(f"S2 failed for {dataset_id}. Terminating...")

        time_taken[dataset_id]["s2"] = toc_s2 - tic_s2

    with open("time_log.yaml", "w") as outf:
        yaml.dump(time_taken, outf)


if __name__ == "__main__":
    main()
