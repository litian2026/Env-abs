import argparse
import math
import os
import pickle
import random

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def sample_interaction(max_interaction, min_interaction=0.0):
    magnitude = np.random.uniform(min_interaction, max_interaction)
    angle = np.random.uniform(-math.pi, math.pi)
    return np.array(
        [magnitude * math.cos(angle), magnitude * math.sin(angle)],
        dtype=np.float32,
    )


FULL_STATE_SCALE = np.array(
    [
        100.0,
        100.0,
        16 * math.pi / 180,
        1.0,
        1.0,
        10.0,
        10.0,
        16 * math.pi / 180,
    ],
    dtype=np.float32,
)


class VelocityPolicyNetwork(nn.Module if nn is not None else object):
    def __init__(self, state_dim, action_dim, action_low, action_high):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
        )
        self.mu = nn.Linear(128, action_dim)
        self.sigma = nn.Linear(128, action_dim)
        self.register_buffer("action_low", torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = self.net(x)
        mu_raw = torch.tanh(self.mu(h))
        mu = self.action_low + (mu_raw + 1.0) / 2.0 * (self.action_high - self.action_low)
        return mu


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, list):
        if len(checkpoint) != 1:
            print(f"Warning: checkpoint list has {len(checkpoint)} entries; using the first one.")
        checkpoint = checkpoint[0]
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format.")

    cleaned = {}
    for key, value in checkpoint.items():
        for prefix in ("module.", "policy.", "agent.policy."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    return cleaned


def infer_policy_state_dim(state_dict):
    for key in ("net.0.weight", "net.0.module.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[1])
    for key, value in state_dict.items():
        if key.endswith("net.0.weight"):
            return int(value.shape[1])
    raise ValueError("Could not infer policy state_dim from checkpoint.")


class StaticPolicyController:
    def __init__(
        self,
        checkpoint_path,
        device,
        action_low,
        action_high,
        v_min,
        v_max,
        action_noise,
    ):
        if torch is None:
            raise ImportError("PyTorch is required for --controller static_policy.")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = extract_state_dict(checkpoint)
        state_dim = infer_policy_state_dim(state_dict)

        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.action_noise = action_noise
        self.policy = VelocityPolicyNetwork(state_dim, 2, self.action_low, self.action_high).to(self.device)
        self.policy.load_state_dict(state_dict, strict=False)
        self.policy.eval()
        self.reset()

    def reset(self, v_max=None):
        if v_max is not None:
            self.v_max = float(v_max)
        self.v = self.v_max
        self.theta = 0.0

    def build_state(self, position, target):
        dx = target[0] - position[0]
        dy = target[1] - position[1]
        if self.state_dim == 8:
            state = np.array(
                [dx, dy, self.v, math.sin(self.theta), math.cos(self.theta), 0.0, 0.0, self.v_max],
                dtype=np.float32,
            )
            return state / FULL_STATE_SCALE

        raise ValueError(
            f"Static perception-data generation expects an 8D OURS/static policy checkpoint, "
            f"but the loaded policy has state_dim={self.state_dim}. The two interaction "
            f"components are set to zero during data generation."
        )

    def command(self, position, target):
        state = torch.tensor(self.build_state(position, target), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            delta = self.policy(state).squeeze(0).detach().cpu().numpy()
        if self.action_noise > 0:
            delta = delta + np.random.normal(0.0, self.action_noise, size=delta.shape)

        self.v = float(np.clip(self.v + delta[0], self.v_min, self.v_max))
        self.theta = wrap_angle(self.theta + delta[1])
        return self.v, self.theta


class ProbingController:
    def __init__(
        self,
        action_low,
        action_high,
        v_min,
        v_max,
        theta_noise,
        pid_gain=0.65,
        pid_phi_gain=0.35,
    ):
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.theta_noise = float(theta_noise)
        self.pid_gain = float(pid_gain)
        self.pid_phi_gain = float(pid_phi_gain)
        self.reset()

    def reset(self, v_max=None):
        if v_max is not None:
            self.v_max = float(v_max)
        self.v = 0.0
        self.theta = 0.0
        self.step = 0

    def command(self, position, target):
        target_angle = math.atan2(target[1] - position[1], target[0] - position[0])
        heading_error = wrap_angle(target_angle - self.theta)

        delta_theta = self.pid_gain * heading_error
        if self.theta_noise > 0:
            delta_theta += np.random.normal(0.0, self.theta_noise)
        delta_theta = float(np.clip(delta_theta, self.action_low[1], self.action_high[1]))

        delta_v = self.pid_phi_gain * (self.v_max - self.v)
        delta_v = float(np.clip(delta_v, self.action_low[0], self.action_high[0]))

        self.v = float(np.clip(self.v + delta_v, self.v_min, self.v_max))
        self.theta = wrap_angle(self.theta + delta_theta)
        self.step += 1

        return self.v, self.theta


def rollout_episode(
    episode_length,
    interaction,
    v_max,
    dt_scale,
    flow_scale,
    arena_size,
    controller,
):
    position = np.random.uniform(-0.25 * arena_size, 0.25 * arena_size, size=2).astype(np.float32)
    target = np.random.uniform(-0.5 * arena_size, 0.5 * arena_size, size=2).astype(np.float32)
    controller.reset(v_max=v_max)

    records = []
    for step in range(episode_length):
        v, theta = controller.command(position, target)

        records.append(
            [
                v,
                math.sin(theta),
                math.cos(theta),
                position[0],
                position[1],
            ]
        )

        position = position.copy()
        position[0] += dt_scale * (v * math.cos(theta) + interaction[0] / flow_scale)
        position[1] += dt_scale * (v * math.sin(theta) - interaction[1] / flow_scale)

    return np.asarray(records, dtype=np.float32)


def windows_from_episode(records, interaction, seq_len, relative_mode):
    features = []
    targets = []

    for end in range(seq_len - 1, len(records)):
        window = records[end + 1 - seq_len:end + 1].copy()
        first_xy = window[0, 3:5].copy()

        if relative_mode == "first_minus_current":
            window[:, 3:5] = first_xy - window[:, 3:5]
        elif relative_mode == "current_minus_first":
            window[:, 3:5] = window[:, 3:5] - first_xy
        elif relative_mode == "absolute":
            pass
        else:
            raise ValueError(f"Unknown relative_mode: {relative_mode}")

        features.append(window)
        targets.append(interaction)

    return features, targets


def generate_dataset(args):
    set_seed(args.seed)

    all_features = []
    all_targets = []
    episode_interactions = []
    episode_v_max = []
    episodes = []

    if args.controller == "static_policy":
        if not args.policy_checkpoint:
            raise ValueError("--policy-checkpoint is required when --controller static_policy.")
        controller = StaticPolicyController(
            checkpoint_path=args.policy_checkpoint,
            device=args.policy_device,
            action_low=[args.delta_v_low, args.delta_theta_low],
            action_high=[args.delta_v_high, args.delta_theta_high],
            v_min=args.v_min,
            v_max=args.v_max,
            action_noise=args.policy_action_noise,
        )
        print(f"Loaded static policy: {args.policy_checkpoint}")
        print(f"Inferred policy state_dim: {controller.state_dim}")
    else:
        controller = ProbingController(
            action_low=[args.delta_v_low, args.delta_theta_low],
            action_high=[args.delta_v_high, args.delta_theta_high],
            v_min=args.v_min,
            v_max=args.v_max,
            theta_noise=args.theta_noise,
            pid_gain=args.probing_pid_gain,
            pid_phi_gain=args.probing_phi_gain,
        )

    for _ in range(args.num_episodes):
        v_max = float(np.random.uniform(args.min_v_max, args.max_v_max))
        max_interaction = args.interaction_ratio * v_max * args.flow_scale
        min_interaction = args.min_interaction_ratio * v_max * args.flow_scale
        interaction = sample_interaction(
            max_interaction=max_interaction,
            min_interaction=min_interaction,
        )

        records = rollout_episode(
            episode_length=args.episode_length,
            interaction=interaction,
            v_max=v_max,
            dt_scale=args.dt_scale,
            flow_scale=args.flow_scale,
            arena_size=args.arena_size,
            controller=controller,
        )

        features, targets = windows_from_episode(
            records=records,
            interaction=interaction,
            seq_len=args.seq_len,
            relative_mode=args.relative_mode,
        )

        all_features.extend(features)
        all_targets.extend(targets)
        episode_interactions.append(interaction)
        episode_v_max.append(v_max)

        episode_targets = np.repeat(interaction.reshape(1, 2), len(records), axis=0)
        episodes.append(np.concatenate([records, episode_targets], axis=1).astype(np.float32))

    dataset = {
        "features": np.asarray(all_features, dtype=np.float32),
        "targets": np.asarray(all_targets, dtype=np.float32),
        "episodes": episodes,
        "episode_interactions": np.asarray(episode_interactions, dtype=np.float32),
        "episode_v_max": np.asarray(episode_v_max, dtype=np.float32),
        "metadata": {
            "feature_names": ["v", "sin_theta", "cos_theta", "relative_x", "relative_y"],
            "target_names": ["interaction_x", "interaction_y"],
            "episode_column_names": [
                "v",
                "sin_theta",
                "cos_theta",
                "x",
                "y",
                "interaction_x",
                "interaction_y",
            ],
            "seq_len": args.seq_len,
            "relative_mode": args.relative_mode,
            "num_episodes": args.num_episodes,
            "episode_length": args.episode_length,
            "min_interaction_ratio": args.min_interaction_ratio,
            "interaction_ratio": args.interaction_ratio,
            "min_v_max": args.min_v_max,
            "max_v_max": args.max_v_max,
            "policy_v_min": args.v_min,
            "dt_scale": args.dt_scale,
            "flow_scale": args.flow_scale,
            "theta_noise": args.theta_noise,
            "arena_size": args.arena_size,
            "controller": args.controller,
            "probing_pid_gain": args.probing_pid_gain,
            "probing_phi_gain": args.probing_phi_gain,
            "policy_checkpoint": args.policy_checkpoint,
            "policy_action_noise": args.policy_action_noise,
            "seed": args.seed,
        },
    }
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate clean 5D motion-history data for interaction perception."
    )
    parser.add_argument("--output", type=str, default="perception_sequences_5d.pkl")
    parser.add_argument("--num-episodes", type=int, default=5000)
    parser.add_argument("--episode-length", type=int, default=80)
    parser.add_argument("--seq-len", type=int, default=15)
    parser.add_argument("--min-interaction-ratio", type=float, default=0.0)
    parser.add_argument("--interaction-ratio", type=float, default=0.8)
    parser.add_argument("--min-v-max", type=float, default=5 * math.pi / 180)
    parser.add_argument("--max-v-max", type=float, default=16 * math.pi / 180)
    parser.add_argument("--dt-scale", type=float, default=10.0)
    parser.add_argument("--flow-scale", type=float, default=6.0)
    parser.add_argument("--theta-noise", type=float, default=0.03)
    parser.add_argument("--arena-size", type=float, default=300.0)
    parser.add_argument("--controller", choices=["probing", "static_policy"], default="probing")
    parser.add_argument("--policy-checkpoint", type=str, default=None)
    parser.add_argument("--policy-device", type=str, default="auto")
    parser.add_argument("--policy-action-noise", type=float, default=0.0)
    parser.add_argument("--delta-v-low", type=float, default=-0.025)
    parser.add_argument("--delta-theta-low", type=float, default=-math.pi / 8)
    parser.add_argument("--delta-v-high", type=float, default=0.025)
    parser.add_argument("--delta-theta-high", type=float, default=math.pi / 8)
    parser.add_argument("--v-min", type=float, default=0.0)
    parser.add_argument("--v-max", type=float, default=16 * math.pi / 180)
    parser.add_argument("--probing-pid-gain", type=float, default=0.65)
    parser.add_argument("--probing-phi-gain", type=float, default=0.35)
    parser.add_argument(
        "--relative-mode",
        choices=["first_minus_current", "current_minus_first", "absolute"],
        default="first_minus_current",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = generate_dataset(args)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "wb") as f:
        pickle.dump(dataset, f)

    print(f"Saved dataset to: {args.output}")
    print(f"features shape: {dataset['features'].shape}")
    print(f"targets shape: {dataset['targets'].shape}")
    print(f"feature names: {dataset['metadata']['feature_names']}")
    print(f"target names: {dataset['metadata']['target_names']}")


if __name__ == "__main__":
    main()