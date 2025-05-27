# MiLoPYP (Pattern Mining (Mi) and particle Localization (Lo) PYthon (PY) Pipeline (P))
<br>

- Key author: Alberto Bartesaghi
- Affiliation: Duke University, Durham, NC, USA
- DOI: https://doi.org/10.1038/s41592-024-02403-6
- This documentation takes content from the [official MiLoPYP tutorial](https://nextpyp.app/milopyp/)  
<br>

## Installation
#### 1. Create a new conda environment [optional, but recommended]:
```
conda create --name MiLoPYP python=3.8
```
And activate the environment.
```
conda activate MiLoPYP
```

#### 2. Clone the cet_pick repo:
```
git clone https://github.com/nextpyp/cet_pick.git
```

#### 3. Install the requirements:
```
pip install -r cet_pick/requirements.txt
```

#### 4. Install PyTorch:
```
pip install torch torchvision torchaudio
```

#### 5. Install cet_pick package and dependencies:
```
pip install -e cet_pick
```
---

#### Download tutorial dataset 
```
wget https://nextpyp.app/files/data/milopyp_globular_tutorial.tbz
tar xvfz milopyp_globular_tutorial.tbz
```
This dataset contains 5 tomograms named as `tilt*.rec`

<br>

## Usage instructions:
There are two steps in their workflow:  
1. Cellular Content Exploration
2. Particle Refined Localization  

I have only used the first (Cellular Content Exploration) step.

#### 1. Prepare input file
You need to make a `.txt` input file. Since we are working with only the reconstructed tomograms, we only need to make the following file:
```
image_name   rec_path

tomo1   path_to_rec_1

tomo2   path_to_rec_2
...
```
where `path_to_rec_1` is the path to a reconstructed tomogram. More than one tomogram can be processed at a time.

***Note:*** *This is a tab separated file.*

***Note:*** *Although tilt-series files are available for tutorial dataset, I am assuming that we don't have access to these as for all the tomograms in our dataset, there are no tilt-series files available*

Place this file in a `data/` directory. Ensure to name the directory `data`. But whenever pointing to this file in the subsequent steps, you don't need to mention `data/`. Note that the scripts will automatically add this to the prefix whenever you open this file from their scripts.


#### 2. Train the model
Again, I used the `3d` mode of the workflow assuming that we don't have access to the tilt-series.
```bash
python cet_pick/cet_pick/simsiam_main.py simsiam3d --num_epochs 20 --exp_id test_sample --bbox 36 --dataset simsiam3d --arch simsiam2d_18 --lr 1e-3 --train_img_txt sample_train_explore_img.txt --batch_size 256 --val_intervals 20 --save_all --gauss 0.8 --dog 3,5 --order xyz
```
where `sample_train_explore_img.txt` is the tab separated file that we made above. I did not tweak with other hyperparameters. You can change the value of `--exp_id` to give a name to your dataset. 


#### 3. Inference step
Again, I used the `3d` mode of the workflow assuming that we don't have access to the tilt-series.
```bash
python cet_pick/cet_pick/simsiam_test_hm_3d.py simsiam3d --exp_id test_sample --bbox 36 --dataset simsiam3d --arch simsiam2d_18 --test_img_txt sample_train_explore_img.txt --load_model exp/simsiam3d/test_sample/model_20.pth --gauss 0.8 --dog 3,5 --order xyz
```

where `sample_train_explore_img.txt` is the tab separated file that we made above. Note that the inference may also be performed on other tomograms using the same trained model.

#### 4. 2D visualization
This step also perform UMAP and clustering. This step is needed to get the coordinates of predicted particles.
```
python cet_pick/cet_pick/plot_2d.py --input exp/simsiam2d3d/test_sample/all_output_info.npz --n_cluster 48 --num_neighbor 40 --mode umap --path exp/simsiam2d3d/test_sample/ --min_dist_vis 1.3e-3 --gpus '-1'
```
***Note:*** *All the output files are stored in a directory called `exp`. This directory will be automatically generated.*


#### 5A. Convert the predicted coordinates to a yaml file similar to one that PickET workflow generates
This is a custom script that I wrote to evaluate coordinates predicted by MiLoPYP using our evaluation scripts.
```bash 
python pickET/accessories/convert_milopyp_preds_to_yaml.py ../milopyp/tomotwin_8tomo_r1/exp/simsiam3d/tomotwin_8r1/all_output_info.npz ../milopyp/tomotwin_8tomo_r1/data/tomotwin_input.txt  /data2/shreyas/mining_tomograms/working/s1_clean_results_picket_v2/tomotwin_8tomos_r1_milopyp_preds/
```
This script takes the output from MiLoPYP as input along with the input .txt file used to run MiLoPYP and the output directory where the newly generated file should be saved. The output is a yaml file in a form similar to the output from PickET. It can then be visualized with the `see_centroids.py` from PickET accessories.

#### 6A. Setting up the environment for the interactive steps
MiLoPYP uses [Arize-AI's Phoenix library](https://docs.arize.com/phoenix) for interactive visualization in 3D. Note that the Phoenix library installed in the MiLoPYP conda environment will likely now work due to some broken Numpy dependencies. I had created a separate `venv` environment for `Phoenix` outside the MiLoPYP conda environment using the following steps:
```bash
conda deactivate
cd ~/Projects/mining_tomograms/environments
mkdir phoenix
python -m venv phoenix
source ~/Projects/mining_tomograms/environments/phoenix/bin/activate
pip install arize-phoenix
pip install arize-phoenix[embeddings]
```
***Note:** Make sure to deactivate the MiLoPYP conda environment and activate the phoenix venv environment before running the interactive steps.*


#### 7. 3D interactive session
##### 1. Load local images
Loading local images generated by `plot_2d.py` in the directory `exp/simsiam3d/test_sample/imgs/` onto a server by running the following command:
```bash
python -m http.server 7000
```
##### 2. Start the interactive session
Deactivate the MiLoPYP conda environment and activate the `phoenix` venv environment made in step 6A by running the following command:
```bash
conda deactivate
source ~/Projects/mining_tomograms/environments/phoenix/bin/activate
```

**Modify the `cet_pick/cet_pick/phoenix_visualization.py` as follows:**  

First, import time module by adding the following line at the top of the script:
```python
import time
```

Then, change line 78 in  from:
```python
    session = px.launch_app(train_ds)
```
to
```python
    print("Launching app")
    session = px.launch_app(train_ds)

    print(f"Phoenix UI launched at: {session.url}")
    print("Press Ctrl+C to stop the server and exit.")

    # Keep the script running indefinitely
    try:
        while True:
            time.sleep(1)  # Sleep for a short duration to prevent busy-waiting
    except KeyboardInterrupt:
        print("\nPhoenix server stopped by user (Ctrl+C detected).")
```

Now, in a new terminal tab, activate the `phoenix` venv environment and start the Phoenix server by running the following command:
```bash
source ~/Projects/mining_tomograms/environments/phoenix/bin/activate
phoenix serve
```

Then, go back to the previous terminal tab and run the following python script to start the interactive session:
```bash
python cet_pick/cet_pick/phoenix_visualization.py --input exp/simsiam3d/test_sample/interactive_info_parquet.gzip
```
In the terminal output, look for the following text: 
```
To view the Phoenix app in you browser, visit http://localhost:xxxx
```
Where, `xxxx` corresponds to the port on which Phoenix app is hosted. Open the link in a browser window.
Follow the instructions on the [MiLoPYP tutorial](https://nextpyp.app/milopyp/explore/#3d-interactive-session) to select embeddings. You can then download the coordinates of the particles corresponding to the selected embeddings in .parquet format.