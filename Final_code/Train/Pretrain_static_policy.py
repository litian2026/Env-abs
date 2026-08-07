import argparse
import glob
import math
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from Algorithm.utils.Network_stage2 import PolicyNetwork
state_scale = np.array([
    100.0,
    100.0,
    16 * math.pi / 180,
    1.0,
    1.0,
    10.0,
    10.0,
    16 * math.pi / 180,
], dtype=np.float32)

def latest_dataset_path():
    pattern = os.path.join(ROOT_DIR, "model_data", "static_expert_*", "static_expert_dataset.npz")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No expert dataset found with pattern: {pattern}")
    return max(paths, key=os.path.getmtime)


def make_output_dir():
    output_dir = os.path.join(
        ROOT_DIR,
        "model_data",
        "pretrain_static_policy_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def scaled_action_mse(mu, actions, action_low, action_high):
    action_scale = (action_high - action_low).clamp_min(1e-8)
    return F.mse_loss((mu - actions) / action_scale, torch.zeros_like(actions))


def train(args):
    dataset_path = args.dataset or latest_dataset_path()
    output_dir = args.output_dir or make_output_dir()

    data = np.load(dataset_path, allow_pickle=True)
    states = torch.tensor(data["states"]/state_scale, dtype=torch.float32)
    actions = torch.tensor(data["actions"], dtype=torch.float32)

    action_low = np.array([-0.025, -math.pi / 8], dtype=np.float32)
    action_high = np.array([0.025, math.pi / 8], dtype=np.float32)
    action_low_tensor = torch.tensor(action_low, dtype=torch.float32)
    action_high_tensor = torch.tensor(action_high, dtype=torch.float32)

    dataset = TensorDataset(states, actions)
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    policy = PolicyNetwork(
        state_dim=8,
        action_dim=2,
        action_low=action_low,
        action_high=action_high,
    ).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_loss = float("inf")
    best_path = os.path.join(output_dir, "model_pretrained_best.pt")
    last_path = os.path.join(output_dir, "model_pretrained_last.pt")

    for epoch in range(1, args.epochs + 1):
        policy.train()
        train_loss = 0.0
        train_count = 0
        for batch_states, batch_actions in train_loader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            mu, _ = policy(batch_states)
            loss = scaled_action_mse(
                mu,
                batch_actions,
                action_low_tensor.to(device),
                action_high_tensor.to(device),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()

            train_loss += loss.item() * batch_states.size(0)
            train_count += batch_states.size(0)

        policy.eval()
        val_loss = 0.0
        val_count = 0
        raw_mae = torch.zeros(2, device=device)
        with torch.no_grad():
            for batch_states, batch_actions in val_loader:
                batch_states = batch_states.to(device)
                batch_actions = batch_actions.to(device)
                mu, _ = policy(batch_states)
                loss = scaled_action_mse(
                    mu,
                    batch_actions,
                    action_low_tensor.to(device),
                    action_high_tensor.to(device),
                )
                val_loss += loss.item() * batch_states.size(0)
                val_count += batch_states.size(0)
                raw_mae += torch.abs(mu - batch_actions).sum(dim=0)

        train_loss /= max(1, train_count)
        val_loss /= max(1, val_count)
        raw_mae = (raw_mae / max(1, val_count)).detach().cpu().numpy()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save([policy.state_dict()], best_path)

        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:04d}/{args.epochs} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"mae_delta_v={raw_mae[0]:.6f} mae_delta_theta={raw_mae[1]:.6f}"
            )

    torch.save([policy.state_dict()], last_path)
    print(f"Dataset: {dataset_path}")
    print(f"Saved best policy: {best_path}")
    print(f"Saved last policy: {last_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Behavior-clone the stage-2 policy from static expert data.")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
