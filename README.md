# V2Xverse-DirectedCP: Communication-Efficient Collaborative Autonomous Driving

[Original V2Xverse Paper](https://arxiv.org/pdf/2404.09496) | [Original Project Page](https://collaborativeperception.github.io/V2Xverse/)

This repository is a research fork of **V2Xverse** for LiDAR-based collaborative autonomous driving in CARLA. It keeps the original end-to-end pipeline for **3D object detection**, **waypoints prediction**, and **closed-loop driving**, and adds a communication-aware collaborative perception stack centered on:

- **Directed-CP style sparse communication**
- **planning-aware request maps** derived from waypoints
- **adaptive communication thresholds** controlled by target communication rate
- **direction-weighted detection loss** for communication-sensitive regions
- **portable repo-relative path management** via `external_paths/...`
- **closed-loop environment validation** before CARLA evaluation

![V2X autonomous driving](simulation/demo/demo.gif)

## Highlights

- **Enhanced CoDriving perception model** in `opencood/models/center_point_codriving.py`
  - integrates direction-aware sparse masking before feature fusion
  - supports planning-guided request maps to bias communication toward route-relevant regions
  - forwards `directed_cp_mask` into the fusion module so communication masks and feature masks stay consistent
- **Adaptive communication control** in `opencood/models/comm_modules/codriving.py`
  - supports round-specific thresholds
  - supports target-rate-based automatic threshold selection
  - supports different request-map radii between perception-only and planning-aware rounds
- **Direction-weighted multiclass loss** in `opencood/loss/direction_weighted_point_pillar_loss.py`
  - extends CenterPoint-style multiclass supervision
  - reweights loss across directional quadrants for communication-oriented training
- **Portable configuration paths**
  - YAML fields starting with `external_paths/` are resolved automatically by both OpenCOOD and planning config loaders
  - this removes hard-coded absolute dataset paths from the default configs
- **Safer closed-loop evaluation workflow**
  - `scripts/check_closed_loop_env.py` validates Python, CARLA egg compatibility, and required dependencies
  - `scripts/eval_driving_e2e.sh` auto-selects a compatible Python interpreter and matching CARLA egg

## Supported Capabilities

- **Tasks**
  - closed-loop collaborative driving
  - LiDAR-based 3D object detection
  - waypoints prediction
- **Collaborative perception methods**
  - enhanced `codriving`
  - `early`
  - `late`
  - `fcooper`
  - `v2xvit`
  - `v2vnet` perception config
  - `single` no-collaboration baseline
- **Modality**
  - LiDAR

## Contents

1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Perception Training and Inference](#perception-training-and-inference)
4. [Planning Training and Evaluation](#planning-training-and-evaluation)
5. [Closed-Loop Evaluation](#closed-loop-evaluation)
6. [Checkpoints](#checkpoints)
7. [Troubleshooting](#troubleshooting)
8. [Acknowledgements](#acknowledgements)
9. [Citation](#citation)

## Installation

### 1. Create the environment

```bash
git clone https://github.com/taipinz/v2xverse_directedcp_official.git
cd v2xverse_directedcp_official
export REPO_ROOT=$PWD

conda create --name v2xverse python=3.7 cmake=3.22.1
conda activate v2xverse

conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install cudnn -c conda-forge

pip install -r opencood/requirements.txt
pip install -r simulation/requirements.txt
pip install -r requirements.txt
```

Notes:

- Closed-loop CARLA evaluation is still tied to the **CARLA 0.9.10.1 Python 3.7 egg**.
- `simulation/requirements.txt` pins `setuptools==41.2.0` because `easy_install` is required for the CARLA egg.

### 2. Download and set up CARLA 0.9.10.1

```bash
chmod +x simulation/setup_carla.sh
./simulation/setup_carla.sh

easy_install carla/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg

mkdir -p external_paths
ln -sfn ${PWD}/carla external_paths/carla_root
```

If you already have a CARLA installation, link it directly:

```bash
mkdir -p external_paths
ln -sfn /path/to/carla external_paths/carla_root
```

To make CARLA deterministic, add the following to `external_paths/carla_root/CarlaUE4/Config/DefaultGameUserSettings.ini`:

```ini
[CARLA/ServerRandomSeed]
Seed = 1234
```

### 3. Install `spconv==1.2.1`

This project uses `spconv 1.2.1` for voxel feature generation in the perception module.

Please follow the upstream installation guide:

https://github.com/traveller59/spconv/tree/v1.2.1

### 4. Set up OpenCOOD

```bash
python setup.py develop
python opencood/utils/setup.py build_ext --inplace
```

### 5. Install `pypcd`

```bash
cd ..
git clone https://github.com/klintan/pypcd.git
cd pypcd
pip install python-lzf
python setup.py install
cd ${REPO_ROOT}
```

### 6. Optional camera dependency

If you plan to use camera-related components such as Lift-Splat-Shoot:

```bash
pip install efficientnet_pytorch==0.7.0
```

## Data Preparation

You can either:

- generate the dataset with the built-in CARLA pipeline, or
- download the original V2Xverse dataset from [Hugging Face](https://huggingface.co/datasets/gjliu/V2Xverse)

This fork expects dataset and CARLA paths to be exposed through `external_paths/`, and the YAML loaders will automatically resolve those paths.

### Link the dataset root

```bash
mkdir -p external_paths
ln -sfn /path/to/dataset_v2xverse external_paths/data_root
```

### Generate a dataset

```bash
cd /path/to/v2xverse_directedcp_official

python simulation/data_collection/init_dir.py --dataset_dir ./dataset
python simulation/data_collection/generate_scripts.py

ln -sfn ${PWD}/dataset external_paths/data_root

CUDA_VISIBLE_DEVICES=0 ./external_paths/carla_root/CarlaUE4.sh --world-port=40000 -prefer-nvidia
CUDA_VISIBLE_DEVICES=1 ./external_paths/carla_root/CarlaUE4.sh --world-port=40002 -prefer-nvidia
CUDA_VISIBLE_DEVICES=2 ./external_paths/carla_root/CarlaUE4.sh --world-port=40004 -prefer-nvidia
...
CUDA_VISIBLE_DEVICES=7 ./external_paths/carla_root/CarlaUE4.sh --world-port=40028 -prefer-nvidia

bash simulation/data_collection/generate_v2xverse_all.sh
```

Generate one route only:

```bash
CUDA_VISIBLE_DEVICES=0 ./external_paths/carla_root/CarlaUE4.sh --world-port=40000 -prefer-nvidia
bash simulation/data_collection/scripts/weather-0/routes_town01_0.sh
```

After generating new data, build the dataset index:

```bash
python simulation/data_collection/gen_index.py
```

The generated index will be saved as `dataset/dataset_index.txt`.

### Dataset structure

```text
weather-0/
  data/
    routes_town{town_id}_{route_id}_w{weather_id}_{datetime}/
      ego_vehicle_{vehicle_id}/
        2d_bbs_{direction}/
        3d_bbs/
        actors_data/
        affordances/
        bev_visibility/
        birdview/
        depth_{direction}/
        env_actors_data/
        lidar/
        lidar_semantic_front/
        measurements/
        rgb_{direction}/
        seg_{direction}/
        topdown/
      rsu_{vehicle_id}/
      log/
  results/
...
weather-13/
```

## Perception Training and Inference

The main perception configs for this fork are:

- `opencood/hypes_yaml/v2xverse/codriving_multiclass_config.yaml`
  - default Directed-CP-enabled CoDriving config
- `opencood/hypes_yaml/v2xverse/codriving_multiclass_config_train.yaml`
  - training-oriented setting with higher communication budget / target rate
- `opencood/hypes_yaml/v2xverse/codriving_multiclass_config_comm.yaml`
  - communication-constrained setting for low-bandwidth experiments

Key additions inside these configs:

- `use_directed_cp: true`
- `directed_cp_args.comm_budget`
- round-specific communication thresholds or target rates
- planning-aware request-map parameters such as `radius_round2`
- `direction_weighted_point_pillar_loss`

### Train perception

Single GPU:

```bash
python opencood/tools/train.py -y opencood/hypes_yaml/v2xverse/codriving_multiclass_config_train.yaml
```

Resume from a checkpoint:

```bash
python opencood/tools/train.py -y opencood/hypes_yaml/v2xverse/codriving_multiclass_config_train.yaml --model_dir ${CHECKPOINT_FOLDER}
```

DDP:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env opencood/tools/train_ddp.py -y opencood/hypes_yaml/v2xverse/codriving_multiclass_config_train.yaml
```

### Inference

Standard evaluation:

```bash
python opencood/tools/inference_multiclass.py --model_dir ${CHECKPOINT_FOLDER}
```

Latency setting:

```bash
python opencood/tools/inference_multiclass_latency.py --model_dir ${CHECKPOINT_FOLDER}
```

Pose-noise setting:

```bash
python opencood/tools/inference_multiclass_w_noise.py --model_dir ${CHECKPOINT_FOLDER}
```

## Planning Training and Evaluation

The planning stack is trained end-to-end on top of a frozen perception checkpoint. In this fork, the default planning config `codriving/hypes_yaml/codriving/end2end_codriving.yaml` now:

- uses `external_paths/data_root/` instead of hard-coded absolute dataset paths
- uses a broader town split
  - train: `1, 2, 3, 4, 6`
  - validation: `7, 8, 10`
  - test: `5`

### Train the planner

```bash
bash scripts/train_planner_e2e.sh ${CUDA_VISIBLE_DEVICES} ${NUM_GPUS} ${PERCEPTION_MODEL_DIR} ${COLLAB_METHOD} ${PLANNER_RESUME}
```

Arguments:

- `CUDA_VISIBLE_DEVICES`: GPU ids, for example `0` or `0,1`
- `NUM_GPUS`: number of processes for distributed training
- `PERCEPTION_MODEL_DIR`: folder containing the perception checkpoint and config
- `COLLAB_METHOD`: one of `codriving`, `early`, `late`, `single`, `fcooper`, `v2xvit`
- `PLANNER_RESUME`: optional planner checkpoint for resuming training

### Evaluate waypoint prediction

```bash
bash scripts/eval_planner_e2e.sh ${CUDA_VISIBLE_DEVICES} ${PERCEPTION_MODEL_DIR} ${COLLAB_METHOD} ${PLANNER_CKPT}
```

Latency evaluation:

```bash
bash scripts/eval_planner_e2e_latency.sh ${CUDA_VISIBLE_DEVICES} ${PERCEPTION_MODEL_DIR} ${COLLAB_METHOD} ${PLANNER_CKPT}
```

Pose-noise evaluation:

```bash
bash scripts/eval_planner_e2e_w_noise.sh ${CUDA_VISIBLE_DEVICES} ${PERCEPTION_MODEL_DIR} ${COLLAB_METHOD} ${PLANNER_CKPT}
```

## Closed-Loop Evaluation

This fork adds an explicit environment validator for closed-loop runs. Use it before launching CARLA evaluation.

### 1. Check the runtime environment

```bash
python scripts/check_closed_loop_env.py
```

If your default `python` is not the CARLA-compatible interpreter, override it:

```bash
export V2XVERSE_PYTHON=$HOME/.local/share/mamba/envs/v2xverse/bin/python
```

`scripts/eval_driving_e2e.sh` will automatically:

- validate the environment
- select a usable Python interpreter
- locate the CARLA egg matching the active Python version

### 2. Launch CARLA

```bash
CUDA_VISIBLE_DEVICES=0 ./external_paths/carla_root/CarlaUE4.sh --world-port=${CARLA_PORT} -prefer-nvidia
```

### 3. Evaluate a single route

```bash
bash scripts/eval_driving_e2e.sh ${ROUTE_ID} ${CARLA_PORT} ${METHOD_TAG} ${REPEAT_ID} ${AGENT_CONFIG} ${SCENARIO_CONFIG}
```

Example:

```bash
bash scripts/eval_driving_e2e.sh 0 40000 codriving 0 codriving_5_10 _1
```

### 4. Evaluate the full benchmark split

The provided batch script partitions the routes into five subsets:

```bash
bash scripts/batch_eval_driving_e2e.sh ${CUDA_DEVICE} ${CARLA_PORT} ${METHOD_TAG} ${REPEAT_ID} ${AGENT_CONFIG} 1s
bash scripts/batch_eval_driving_e2e.sh ${CUDA_DEVICE} ${CARLA_PORT} ${METHOD_TAG} ${REPEAT_ID} ${AGENT_CONFIG} 2s
bash scripts/batch_eval_driving_e2e.sh ${CUDA_DEVICE} ${CARLA_PORT} ${METHOD_TAG} ${REPEAT_ID} ${AGENT_CONFIG} 3s
bash scripts/batch_eval_driving_e2e.sh ${CUDA_DEVICE} ${CARLA_PORT} ${METHOD_TAG} ${REPEAT_ID} ${AGENT_CONFIG} 4s
bash scripts/batch_eval_driving_e2e.sh ${CUDA_DEVICE} ${CARLA_PORT} ${METHOD_TAG} ${REPEAT_ID} ${AGENT_CONFIG} 5s
```

### Closed-loop arguments

- `ROUTE_ID`
  - corresponds to `simulation/leaderboard/data/evaluation_routes/town05_short_r${ROUTE_ID}.xml`
- `CARLA_PORT`
  - must match the `--world-port` used when launching CARLA
- `METHOD_TAG`
  - experiment tag used in the saved result folder
- `REPEAT_ID`
  - repeat index for the current run
- `AGENT_CONFIG`
  - mapped to `simulation/leaderboard/team_code/agent_config/pnp_config_${AGENT_CONFIG}.yaml`
  - edit the perception and planner checkpoint paths inside the config before evaluation
- `SCENARIO_CONFIG`
  - mapped to `simulation/leaderboard/leaderboard/scenarios/scenario_parameter${SCENARIO_CONFIG}.yaml`
  - the provided batch script uses `_1` to `_5`

## Checkpoints

You can reuse the original V2Xverse public assets and then fine-tune the modified configs in this fork.

Original resources:

- dataset: https://huggingface.co/datasets/gjliu/V2Xverse
- checkpoints: https://huggingface.co/gjliu/v2xverse

Recommended local layout:

```text
checkpoints/
  codriving/
    perception/
    planner/
  early_fusion/
    perception/
    planner/
  late_fusion/
    perception/
    planner/
```

If you use closed-loop evaluation, update the files under `simulation/leaderboard/team_code/agent_config/` so that:

- `perception.perception_model_dir` points to your perception checkpoint folder
- `planning.planner_model_checkpoint` points to your planner checkpoint file

## Troubleshooting

### CARLA egg mismatch

If `scripts/check_closed_loop_env.py` reports that no CARLA egg matches the current Python:

- activate a Python 3.7 environment, or
- export `V2XVERSE_PYTHON` to a Python 3.7 interpreter that already has the required packages installed

### Path issues in YAML configs

This fork resolves paths beginning with `external_paths/` automatically. Prefer symlinks such as:

```bash
external_paths/carla_root -> /path/to/carla
external_paths/data_root  -> /path/to/dataset_v2xverse
```

instead of editing multiple absolute paths in the YAML files.

### Shut down simulation on Linux

If CARLA processes hang, stop them manually:

```bash
ps U ${USER} | grep -E 'python|carla'
kill -9 ${PID}
pkill -u ${USER} -f carla
```

## Acknowledgements

This implementation is based on code from several repositories.

- [CARLA leaderboard](https://github.com/carla-simulator/leaderboard)
- [Scenario runner](https://github.com/carla-simulator/scenario_runner)
- [Interfuser](https://github.com/opendilab/InterFuser)
- [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD)
- [HEAL](https://github.com/yifanlu0227/HEAL)
- [v2xverse](https://github.com/CollaborativePerception/V2Xverse)

## Citation

If you use this repository, please cite the original V2Xverse paper:

```bibtex
@article{liu2024codriving,
  title={Towards Collaborative Autonomous Driving: Simulation Platform and End-to-End System},
  author={Liu, Genjia and Hu, Yue and Xu, Chenxin and Mao, Weibo and Ge, Junhao and Huang, Zhengxiang and Lu, Yifan and Xu, Yinda and Xia, Junkai and Wang, Yafei and others},
  journal={arXiv preprint arXiv:2404.09496},
  year={2024}
}
```
