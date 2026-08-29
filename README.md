# Source Code for Interaction-Based Environmental Abstraction

This repository contains the source code associated with the manuscript:

> **Environmental abstraction from interaction enables microrobot navigation in physically complex environments**

## Overview

Microrobots have limited capacity for onboard environmental sensing, yet local
physical conditions can continually alter the relationship between actuation and
realised motion. This work uses recent action-response histories as an alternative
source of navigation-relevant environmental information. A temporal encoder learns
a two-dimensional interaction cue that summarises how local physical interactions
affect microrobot motion. The cue then conditions a reinforcement-learning
navigation policy.

The released code contains two complementary modules:

1. **Interaction-cue learning** generates simulation-labelled action-response
   sequences and trains the temporal interaction encoder.
2. **Navigation-policy learning** trains the cue-conditioned policy through expert
   pretraining, static-context reinforcement learning, and dynamic-context
   reinforcement learning.

These modules are provided separately to reflect their distinct training stages.
The interaction-cue labels used during policy training are available directly in
simulation, whereas the trained temporal encoder estimates the cue from recent
action-response histories for physical deployment.

## Repository structure

```text
Env-abs/
|-- Interaction cue/
|   |-- Generate_perception_data_new.py
|   `-- Model_estimation_patt.py
|-- RL_code_all/
|   |-- Algorithm/
|   |   |-- Image_process.py
|   |   |-- PathFollowing.py
|   |   |-- PPO.py
|   |   `-- utils/
|   |       |-- networks.py
|   |       `-- ppo_agent.py
|   |-- SwarmEnv/
|   |   `-- RLEnv.py
|   |-- Train/
|   |   |-- Generate_static_expert_dataset.py
|   |   |-- Pretrain_static_policy.py
|   |   `-- Train_navigation_policy.py
|   `-- README.md
`-- README.md
```

## Software requirements

The code was developed in Python. Python 3.10 is recommended. The principal
dependencies are:

```text
numpy
torch
gym
stable-baselines3
matplotlib
opencv-python
scikit-learn
joblib
sympy
```

A suitable Conda environment can be created with:

```bash
conda create -n microrobot-navigation python=3.10
conda activate microrobot-navigation
pip install numpy torch gym stable-baselines3 matplotlib opencv-python scikit-learn joblib sympy
```

Run each command from the directory containing the corresponding script unless
stated otherwise.

## 1. Learning the interaction cue

The interaction-cue module first generates labelled motion histories in simulation
and then trains an attention-based temporal encoder. Each input sequence contains
control and motion information over a short temporal window. The regression target
is the two-dimensional interaction variable used to generate that simulated
trajectory.

### Generate simulation-labelled action-response histories

From `Interaction cue/`, run:

```bash
python Generate_perception_data_new.py \
  --controller probing \
  --num-episodes 5000 \
  --episode-length 80 \
  --seq-len 15 \
  --min-interaction-ratio 0.0 \
  --interaction-ratio 0.8 \
  --output perception_sequences_static_policy_5d.pkl
```

The cue magnitude is sampled within the range controlled by
`--min-interaction-ratio` and `--interaction-ratio`; its direction is sampled in
two dimensions. The resulting dataset contains:

```text
Input features: [v, sin(theta), cos(theta), relative_x, relative_y]
Targets:        [interaction_x, interaction_y]
```

The data generator can alternatively use a pretrained static navigation policy as
the controller:

```bash
python Generate_perception_data_new.py \
  --controller static_policy \
  --policy-checkpoint PATH/TO/STATIC_POLICY.pt \
  --output perception_sequences_static_policy_5d.pkl
```

### Train the temporal interaction encoder

Keep `perception_sequences_static_policy_5d.pkl` in `Interaction cue/` and run:

```bash
python Model_estimation_patt.py
```

The encoder uses separate self-attention branches for the control-input and
robot-state sequences, followed by cross-attention and gated feature fusion. It
predicts the interaction cue as a two-dimensional vector.

The script saves the trained model, preprocessing scalers, configuration, and
training curve using the `wind_estimation_v1` and `supplementary_figures` output
names defined in the script.

## 2. Training the navigation policy

Run the following commands from `RL_code_all/`. The proposed policy receives the
state:

```text
[dx, dy, v, sin(theta), cos(theta), e_x, e_y, v_max]
```

Here, `dx` and `dy` are the displacement to the current target, `v` and `theta`
describe the current motion state, `(e_x, e_y)` is the interaction cue, and
`v_max` is the platform velocity bound.

Three policy configurations are implemented:

- `ours`: cue-conditioned policy using the complete state;
- `no_env`: policy without the interaction-cue components; and
- `history`: policy receiving a temporal history of the reduced state.

### Stage 1: expert-data generation and behaviour-cloning pretraining

Generate expert trajectories in the static simulation environment:

```bash
python Train/Generate_static_expert_dataset.py \
  --output-dir model_data/static_expert
```

Pretrain the proposed policy:

```bash
python Train/Pretrain_static_policy.py \
  --mode ours \
  --dataset model_data/static_expert/static_expert_dataset.npz \
  --output-dir model_data/pretrain_ours
```

The same dataset can be used to pretrain the No-env baseline by changing
`--mode ours` to `--mode no_env`.

### Stage 2: reinforcement learning in static contexts

```bash
python Train/Train_navigation_policy.py \
  --mode ours \
  --phase static \
  --pretrained-dir model_data/pretrain_ours \
  --actor-name model_pretrained_best.pt \
  --output-dir model_data/ours_static
```

### Stage 3: reinforcement learning in dynamic interaction contexts

Use the actor and critic checkpoints produced by the static-context stage:

```bash
python Train/Train_navigation_policy.py \
  --mode ours \
  --phase dynamic \
  --pretrained-dir model_data/ours_static \
  --actor-name MODEL_CHECKPOINT.pt \
  --critic-name CRITIC_CHECKPOINT.pt \
  --output-dir model_data/ours_dynamic
```

Set `--mode no_env` or `--mode history` to train the corresponding baseline.
Use `--seed` for an individual random seed and `--cpu` to disable CUDA. Full PPO
training uses parallel environments and can be computationally expensive.

## Outputs

Depending on the selected stage, the scripts generate:

- labelled interaction-cue datasets (`.pkl`);
- trained interaction encoders and preprocessing scalers;
- expert trajectory datasets (`.npz`);
- policy and critic checkpoints (`.pt`);
- episode-level training records (`training_log.npz`); and
- training-loss and reward figures.

Generated datasets and model checkpoints are not included in the source tree
unless distributed separately.

## Reproducibility notes

- The cue-estimation and policy-training modules are currently run independently;
  this release does not provide a single end-to-end deployment script.
- Policy learning uses the simulation-defined interaction variable as the cue. The
  perception module learns to recover the same variable from action-response
  histories.
- Full reproduction requires the trained checkpoints and the computational budget
  specified for the corresponding experiments.
- Default paths and checkpoint names in the scripts can be overridden through the
  command-line arguments shown above.

## Citation

If you use this code, please cite the associated manuscript. Full bibliographic
information will be added after publication.
