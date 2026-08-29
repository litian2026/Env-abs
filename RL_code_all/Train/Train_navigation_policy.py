import argparse
from html import parser
import math
import os
import random
import sys
from collections import deque
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from matplotlib import pyplot as plt
from multiprocessing import freeze_support
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from torch.optim.lr_scheduler import LambdaLR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.PPO import PPO1
from Algorithm.PathFollowing import PathFollowing
from SwarmEnv.RLEnv import CustomEnv


FULL_STATE_SCALE = np.array([
    100.0,
    100.0,
    16 * math.pi / 180,
    1.0,
    1.0,
    10.0,
    10.0,
    16 * math.pi / 180,
], dtype=np.float32)
NO_ENV_IDX = np.array([0, 1, 2, 3, 4, 7], dtype=np.int64)
NO_ENV_STATE_SCALE = FULL_STATE_SCALE[NO_ENV_IDX]
ACTION_SCALE = np.array([0.025, math.pi / 8], dtype=np.float32)
HISTORY_FEATURE_DIM = 6


def make_env(env_id, rank, seed=0):
    def _thunk():
        env = CustomEnv(worker_id=rank + 153)
        return env
    return _thunk


class TrainingConfig:
    num_envs = 18
    episode_num = 2500100
    episode_length = 250
    dim_out_policy = 2
    epsilon = 0.2
    action_low = np.array([-0.025, -math.pi / 8], dtype=np.float32)
    action_high = np.array([0.025, math.pi / 8], dtype=np.float32)
    state_range = np.array([[0, -2 * math.pi], [16 * math.pi / 180, 2 * math.pi]], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Train comparison policies for stage-2 navigation.")
    parser.add_argument("--mode", choices=["ours", "no_env", "history"], default="history")
    parser.add_argument("--phase", choices=["static", "dynamic"], default="dynamic")
    parser.add_argument("--flow-ratio", type=float, default=0.7)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-envs", type=int, default=TrainingConfig.num_envs)
    parser.add_argument("--episode-num", type=int, default=TrainingConfig.episode_num)
    parser.add_argument("--episode-length", type=int, default=TrainingConfig.episode_length)
    parser.add_argument("--history-window", type=int, default=15)
    parser.add_argument("--pretrained-dir", type=str, default=r"./model_data/results_history_2026-05-09_18-57-32")
    parser.add_argument("--actor-name", type=str, default="model34000.pt")
    parser.add_argument("--critic-name", type=str, default="model_critic34000.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--update-size", type=int, default=1024)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_zero_history(window):
    return deque([np.zeros(HISTORY_FEATURE_DIM, dtype=np.float32) for _ in range(window)], maxlen=window)


def no_env_state(obs):
    return obs[NO_ENV_IDX]


def normalize_no_env(obs):
    return no_env_state(obs).astype(np.float32) / NO_ENV_STATE_SCALE


def normalize_action(action):
    return np.asarray(action, dtype=np.float32) / ACTION_SCALE


def history_entry(obs, action = None):
    return normalize_no_env(obs).astype(np.float32)


def build_policy_state(obs, mode, history_buffer=None):
    obs = np.asarray(obs, dtype=np.float32)
    if mode == "ours":
        return obs / FULL_STATE_SCALE
    if mode == "no_env":
        return normalize_no_env(obs)
    if mode == "history":
        current = normalize_no_env(obs)
        history = np.concatenate(list(history_buffer), dtype=np.float32)
        return np.concatenate([current, history], dtype=np.float32)
    raise ValueError(f"Unknown mode: {mode}")


def mode_dimensions(mode, history_window):
    if mode == "ours":
        return 8, "mlp", None, 0
    if mode == "no_env":
        return 6, "mlp", None, 0
    if mode == "history":
        history_dim = history_window * HISTORY_FEATURE_DIM
        return 6 + history_dim, "history", 6, history_dim
    raise ValueError(f"Unknown mode: {mode}")


def load_checkpoint_if_requested(ppo, args, device):
    if not args.pretrained_dir:
        print("No pretrained checkpoint specified; training from current initialization.")
        return

    actor_name = args.actor_name or "model_pretrained_best.pt"
    actor_path = os.path.join(args.pretrained_dir, actor_name)
    if os.path.exists(actor_path):
        actor_params = torch.load(actor_path, map_location=device)
        for actor_parameter in actor_params:
            ppo.agent.policy.load_state_dict(actor_parameter, strict= True)
            ppo.agent.target_policy.load_state_dict(actor_parameter, strict=True)
        print(f"Loaded actor checkpoint: {actor_path}")
    else:
        print(f"Actor checkpoint not found, skipped: {actor_path}")

    if args.critic_name:
        critic_path = os.path.join(args.pretrained_dir, args.critic_name)
        if os.path.exists(critic_path):
            critic_params = torch.load(critic_path, map_location=device)
            for critic_parameter in critic_params:
                ppo.agent.critic.load_state_dict(critic_parameter, strict=False)
                ppo.agent.target_critic.load_state_dict(critic_parameter, strict=False)
            print(f"Loaded critic checkpoint: {critic_path}")
        else:
            print(f"Critic checkpoint not found, skipped: {critic_path}")


def running_mean(values, window=25):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return values
    out = np.zeros_like(values)
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out[i] = np.mean(values[lo:i + 1])
    return out


def save_training_log(res_dir, current_episode, total_reward, episode_dis, args):
    np.savez_compressed(
        os.path.join(res_dir, "training_log.npz"),
        episode=np.arange(1, current_episode + 1, dtype=np.int32),
        reward=total_reward[:current_episode],
        episode_dis=episode_dis[:current_episode],
        mode=np.asarray(args.mode),
        seed=np.asarray(args.seed),
        phase=np.asarray(args.phase),
        flow_ratio=np.asarray(args.flow_ratio),
    )
def save_training_plots(res_dir, current_episode, total_reward, loss_pi_rec, loss_v_rec):
    x = range(1, current_episode + 1)
    fig, ax = plt.subplots()
    ax.plot(x, running_mean(total_reward[:current_episode]), label="reward")
    ax.legend()
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    title = f"Reward_over_episode{current_episode}"
    ax.set_title(title)
    plt.savefig(os.path.join(res_dir, title + ".png"))
    plt.close(fig)

    loss_pi = [item for sublist in loss_pi_rec for item in sublist]
    if len(loss_pi) > 1:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(loss_pi) + 1), loss_pi, label="policy_loss")
        ax.legend()
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        title = f"Policy-loss{current_episode}"
        ax.set_title(title)
        plt.savefig(os.path.join(res_dir, title + ".png"))
        plt.close(fig)

    loss_v = [item for sublist in loss_v_rec for item in sublist]
    if len(loss_v) > 1:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(loss_v) + 1), loss_v, label="value_loss")
        ax.legend()
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        title = f"Value_loss{current_episode}"
        ax.set_title(title)
        plt.savefig(os.path.join(res_dir, title + ".png"))
        plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)

    config = TrainingConfig
    config.num_envs = args.num_envs
    config.episode_num = args.episode_num
    config.episode_length = args.episode_length

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dim_input, policy_kind, current_state_dim, history_dim = mode_dimensions(args.mode, args.history_window)

    envs = SubprocVecEnv([make_env(i, i) for i in range(config.num_envs)])
    pf = PathFollowing(config.num_envs)
    pf.reference_points = [pf.generate_trajectory("line", 100) for _ in range(config.num_envs)]
    for index, reference_points in enumerate(pf.reference_points):
        envs.env_method("set_reference_points", reference_points, indices=[index])
    envs.env_method("set_episode_length", config.episode_length)
    envs.env_method("set_v_max", 16 * math.pi / 180)
    envs.env_method("set_flow_x", 0.0)

    robot_pos = np.ones([config.num_envs, 2], dtype=np.float32) * 300
    for index in range(config.num_envs):
        current_goal = pf.getCurrentPoint(robot_pos[index], index)
        envs.env_method("set_current_goal", current_goal, indices=[index])

    ppo = PPO1(
        dim_input,
        config.dim_out_policy,
        dim_input,
        config.epsilon,
        device,
        
        config.action_low,
        config.action_high,
        config.state_range,
        K_epochs=5,
        hidden_dim=64,
        gamma=0.99,
        tau=0.01,
        lr_actor=1e-4,
        lr_critic=1e-3,
        Lambda=0.95,
        policy_kind=policy_kind,
        current_state_dim=current_state_dim,
        history_dim=history_dim,
        history_latent_dim=2,
        
        history_window=args.history_window,
        history_feature_dim=HISTORY_FEATURE_DIM,
    )
    load_checkpoint_if_requested(ppo, args, device)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_res_dir = os.path.join("model_data", f"results_{args.mode}_{args.phase}_{current_time}")
    res_dir = args.output_dir or default_res_dir
    os.makedirs(res_dir, exist_ok=True)

    scheduler = LambdaLR(ppo.agent.policy_optimizer, lr_lambda=lambda step: 1.0)
    obs = envs.reset()
    cycle_step = np.zeros(config.num_envs, dtype=np.int32)
    last_theta = np.zeros(config.num_envs, dtype=np.float32)
    last_psi = np.zeros(config.num_envs, dtype=np.float32)
    history_buffers = [make_zero_history(args.history_window) for _ in range(config.num_envs)]
    memory = [[] for _ in range(config.num_envs)]

    total_reward = np.zeros(config.episode_num, dtype=np.float32)
    episode_dis = np.zeros(config.episode_num, dtype=np.float32)
    episode_reward = np.zeros([config.num_envs, config.episode_length], dtype=np.float32)
    loss_pi_rec = []
    loss_v_rec = []
    current_episode = 0
    symbol = False

    for global_step in range(config.episode_length * config.episode_num):
        for index in range(config.num_envs):
            pf.t[index] += 1
            cycle_step[index] += 1

        policy_states = []
        raw_actions = []
        theta_list = []
        psi_list = []
        env_actions = []
        for index in range(config.num_envs):
            policy_state = build_policy_state(obs[index], args.mode, history_buffers[index])
            action, theta, psi = ppo.select_action(policy_state, last_theta[index], last_psi[index])
            policy_states.append(policy_state)
            raw_actions.append(action)
            theta_list.append(theta)
            psi_list.append(psi)
            env_actions.append(np.concatenate([action, np.array([theta, psi, cycle_step[index]], dtype=np.float32)]))

        next_obs, rewards, dones, infos = envs.step(env_actions)
        robot_pos = np.array([info["pos"] for info in infos])
        done_info = np.array([info["dones"] for info in infos], dtype=bool)
        dis = np.array([info["dis"] for info in infos], dtype=np.float32)
        episode_reward[:, global_step % config.episode_length] = rewards

        next_policy_states = []
        for index in range(config.num_envs):
            if args.mode == "history":
                history_buffers[index].append(history_entry(obs[index], raw_actions[index]))
            next_policy_states.append(build_policy_state(next_obs[index], args.mode, history_buffers[index]))
            memory[index].append([policy_states[index], raw_actions[index], rewards[index], next_policy_states[index], done_info[index]])

        obs = next_obs
        last_theta = np.asarray(theta_list, dtype=np.float32)
        last_psi = np.asarray(psi_list, dtype=np.float32)

        if len(ppo.memory) >= args.update_size:
            loss_policy, loss_value = ppo.update()
            loss_pi_rec.append(loss_policy)
            loss_v_rec.append(loss_value)
            memory = [[] for _ in range(config.num_envs)]

        for index, done in enumerate(done_info):
            if not done:
                continue

            adv = 0.0
            temp = []
            for item in reversed(memory[index]):
                state = torch.from_numpy(item[0]).float().to(device)
                next_state = torch.from_numpy(item[3]).float().to(device)
                value_s = ppo.agent.critic(state).item()
                value_s_ = ppo.agent.critic(next_state).item()
                td_error = item[2] + ppo.gamma * value_s_ * (1 - item[4]) - value_s
                adv = adv * ppo.gamma * ppo.Lambda + td_error
                temp.append([item[0], item[1], item[2], item[3], item[4], [adv]])
            for item in reversed(temp):
                ppo.push_data(item)

            env_cumulative_rewards = episode_reward[index, :].sum()
            total_reward[current_episode] = max(env_cumulative_rewards, -500)
            episode_dis[current_episode] = dis[index]
            episode_reward[index, :] = 0
            current_episode += 1

            reference_points = pf.generate_trajectory("line", 100)
            pf.reference_points[index] = reference_points
            envs.env_method("set_reference_points", reference_points, indices=[index])
            current_goal = pf.getCurrentPoint(robot_pos[index], index)
            envs.env_method("set_current_goal", current_goal, indices=[index])
            pf.t[index] = 0
            memory[index] = []
            cycle_step[index] = 0
            history_buffers[index] = make_zero_history(args.history_window)

            random_v_max = np.random.uniform(5 * math.pi / 180, 16 * math.pi / 180)
            if args.phase == "static":
                v_flow = 0.0
            else:
                v_flow = random_v_max * args.flow_ratio * 5.0
            envs.env_method("set_v_max", float(random_v_max), indices=[index])
            envs.env_method("set_flow_x", float(v_flow), indices=[index])
            #obs[index, :] = envs.env_method("reset", indices=[index])[0]
            obs= envs.reset()
            last_theta[index] = 0
            last_psi[index] = 0
            scheduler.step()
            symbol = True
            print(f"Mode {args.mode} | Episode {current_episode} | LR {scheduler.get_last_lr()[0]:.2e} | v_max {random_v_max * 180 / math.pi:.2f}")

            if current_episode >= config.episode_num:
                break

        if symbol and current_episode > 0 and current_episode % args.save_interval == 0:
            torch.save([ppo.agent.policy.state_dict()], os.path.join(res_dir, f"model{current_episode}.pt"))
            torch.save([ppo.agent.critic.state_dict()], os.path.join(res_dir, f"model_critic{current_episode}.pt"))
            np.save(os.path.join(res_dir, "rewards.npy"), total_reward)
            np.save(os.path.join(res_dir, "episode_dis.npy"), episode_dis)
            save_training_log(res_dir, current_episode, total_reward, episode_dis, args)
            save_training_plots(res_dir, current_episode, total_reward, loss_pi_rec, loss_v_rec)
            symbol = False

        if current_episode >= config.episode_num:
            break

    torch.save([ppo.agent.policy.state_dict()], os.path.join(res_dir, f"model{current_episode}.pt"))
    torch.save([ppo.agent.critic.state_dict()], os.path.join(res_dir, f"model_critic{current_episode}.pt"))
    np.save(os.path.join(res_dir, "rewards.npy"), total_reward)
    np.save(os.path.join(res_dir, "episode_dis.npy"), episode_dis)
    save_training_log(res_dir, current_episode, total_reward, episode_dis, args)
    print(f"Training complete. Saved to {res_dir}")


if __name__ == "__main__":
    freeze_support()
    main()
