# Instruction for Replica Semantic Accuracy Evaluation

### Ground Truth PointClouds
We use the same semantically labeled groundtruth pointclouds as in ConceptGraphs for the evaluations of the semantic accuracy in the paper, which can be obtained through the following [Google Drive Link](https://drive.google.com/file/d/1NhQIM5PCH5L5vkZDSRq6YF1bRaSX2aem/view).

### Dataset Generation
To be able to use the Replica Dataset in a stereo camera setup (as required by FindAnything), we used the iMAP version of the replica sequences and rendered an additional second image with a baseline of 6cm. The script to produce the dataset is provided in `[okvis2x_base_directory]/tools/replica-imap-stereo.sh`.

#### Setup
1. Create a conda environment as follows:

    ``` sh
    conda create -n replica-imap-stereo python=3.8 cmake=3.14
    conda activate replica-imap-stereo
    conda install habitat-sim=0.2.2 -c conda-forge -c aihabitat
    ```
2. Download the Replica dataset (31.6 GB download + 42.3 GB to extract) using [this script](https://raw.githubusercontent.com/facebookresearch/Replica-Dataset/refs/heads/main/download.sh).
3. Download the iMAP Replica sequences (11.6 GB download + 11.8 GB to extract) [from here](https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip).

#### Usage
``` text
replica-imap-stereo.py REPLICA_DIR IMAP_DIR OUT_DIR [SCENE] ...
```
Run the script with the -h option to display more detailed usage help. 

### Dependencies
* [ConceptGraphs](https://github.com/concept-graphs/concept-graphs): Our evaluations follow the original ConceptGraphs Evaluation setup, so some of the code is re-used from there
* [ChamferDist](https://github.com/krrish94/chamferdist): ConceptGraphs Evaluation functions use some of the functionality of the ChamferDist package. Please install chamferdist via the setup.py of this directory.
* The files `pointclouds.py`, `projutils.py` and `structutils.py` are taken from the [GradSLAM](https://github.com/gradslam/gradslam/tree/main/gradslam/structures) package (easier than getting in the whole package as a dependency).
* Install the packages required to run the evaluation experiment by using the `requirements.txt` here provided
To download the tested commits of aboves package, please use the provided script `setup_eval_dependencies.sh` (assuming your git configurations are properly set up). Please remember to unpack `text_queries.zip` onto the same level.

### Run Evaluation
* `--replica_seq_path`: Path to generated Dataset
* `--gt-path`: Path to Ground Truth Point Clouds and Trajectory
* `--results_path`: Base Folder of FindAnything generated Outputs
* `--query_path`: Path where Queries (CLIP embedding of category as a txt file) are stored. For replica we provide them in `text_queries.zip`
* `--n_exclude`: Number of classes to be excluded (following ConceptGraph logic, default: 6)
* `--debug`: Runs some Open3D based visualization of predicted class for debugging (Ground Truth / Predicted can be toggled by pressing `1`/`2`)


Example:
```bash
python3 replica_eval_sem.py --replica_seq_path /path/to/generated/sequence/ --gt_path /path/to/gt_pcl_and_traj/ --results_path /path/to/results_base/ --query_path ./text_queries/
```

The results folder should at least contain the following files and structure (with N being the ID of the last submap):
```md
  [results_base]
  ├── office0
  │   ├── objectMap1.csv
  │   ├── ...
  │   ├── objectMap[N].csv
  │   ├── mesh_kf000001.ply
  │   ├── ...
  │   ├── mesh_kf[N].ply
  │   └── okvis2-slam-final_trajectory.csv
  ├── ...
  ├── office4
  ├── room0
  ├── ...
  └── room2
     └── ...
  ```
