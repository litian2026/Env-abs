import argparse
import math
import os
from datetime import datetime

import numpy as np


def wrap_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def make_line_goal(rng, origin):
    # Match the old training style: targets are placed in one of four distant quadrants.
    dx = rng.uniform(60, 100) * rng.choice([-1, 1])
    dy = rng.uniform(60, 100) * rng.choice([-1, 1])
    return origin + np.array([dx, dy], dtype=np.float32)


def expert_action(state, action_low, action_high, k_v):
    dx, dy, v, sin_theta, cos_theta, _, _, v_max = state
    theta = math.atan2(sin_theta, cos_theta)
    theta_target = math.atan2(dy, dx)
    distance = math.sqrt(dx * dx + dy * dy)

    heading_error = wrap_angle(theta_target - theta)
    closing_speed = v * math.cos(heading_error)

    v_target = k_v * distance - 0 * closing_speed
    v_target = float(np.clip(v_target, 0.0, v_max))

    delta_v = v_target - v
    delta_theta = wrap_angle(theta_target - theta)

    action = np.array([delta_v, delta_theta], dtype=np.float32)
    return np.clip(action, action_low, action_high)


def transition(position, v, theta, action, goal, v_max, dt_scale, xi_x=0.0, xi_y=0.0):
    delta_v, delta_theta = action
    v_next = float(np.clip(v + delta_v, 0.0, v_max))
    theta_next = wrap_angle(theta + float(delta_theta))

    # Match SwarmEnv/RLEnv.py::DynamicsEnv.step exactly:
    # x <- x + 10 * (v*cos(theta) + xi_x/6)
    # y <- y + 10 * (v*sin(theta) - xi_y/6)
    next_position = np.array([
        position[0] + dt_scale * (v_next * math.cos(theta_next) + xi_x / 6.0),
        position[1] + dt_scale * (v_next * math.sin(theta_next) - xi_y / 6.0),
    ], dtype=np.float32)
    diff = goal - next_position
    next_state = np.array([
        diff[0],
        diff[1],
        v_next,
        math.sin(theta_next),
        math.cos(theta_next),
        xi_x,
        xi_y,
        v_max,
    ], dtype=np.float32)
    return next_position, v_next, theta_next, next_state


def collect_dataset(args):
    rng = np.random.default_rng(args.seed)
    action_low = np.array([-args.delta_v_limit, -args.delta_theta_limit], dtype=np.float32)
    action_high = np.array([args.delta_v_limit, args.delta_theta_limit], dtype=np.float32)
    origin = np.array([args.origin_x, args.origin_y], dtype=np.float32)

    states = []
    actions = []
    next_states = []
    rewards = []
    dones = []
    goals = []
    positions = []
    episode_ids = []
    step_ids = []

    for episode in range(args.episodes):
        position = origin.copy()
        goal = make_line_goal(rng, origin)
        v_max = float(rng.uniform(args.v_max_low, args.v_max_high))
        v = 0.0
        theta = 0.0
        arrived_steps = 0

        diff = goal - position
        state = np.array([
            diff[0],
            diff[1],
            v,
            math.sin(theta),
            math.cos(theta),
            0.0,
            0.0,
            v_max,
        ], dtype=np.float32)

        for step in range(args.episode_length):
            action = expert_action(state, action_low, action_high, args.k_v)
            next_position, v, theta, next_state = transition(
                position,
                v,
                theta,
                action,
                goal,
                v_max,
                args.dt_scale,
                xi_x=0.0,
                xi_y=0.0,
            )

            distance = float(math.sqrt(next_state[0] ** 2 + next_state[1] ** 2))
            reward = -args.distance_weight * distance
            done = False
            if distance < args.arrive_radius:
                arrived_steps += 1
            else:
                arrived_steps = 0
            if arrived_steps >= args.arrive_hold_steps:
                done = True
            if step == args.episode_length - 1:
                done = True

            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            goals.append(goal.copy())
            positions.append(position.copy())
            episode_ids.append(episode)
            step_ids.append(step)

            position = next_position
            state = next_state
            if done:
                break

        if (episode + 1) % args.log_interval == 0:
            print(f"Collected {episode + 1}/{args.episodes} episodes, {len(states)} transitions")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(
            "model_data",
            "static_expert_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "static_expert_dataset.npz")
    np.savez_compressed(
        output_path,
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        next_states=np.asarray(next_states, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.bool_),
        goals=np.asarray(goals, dtype=np.float32),
        positions=np.asarray(positions, dtype=np.float32),
        episode_ids=np.asarray(episode_ids, dtype=np.int32),
        step_ids=np.asarray(step_ids, dtype=np.int32),
        state_description=np.asarray([
            "dx", "dy", "v", "sin(theta)", "cos(theta)", "xi_x", "xi_y", "v_max"
        ]),
        action_description=np.asarray(["delta_v", "delta_theta"]),
    )
    print(f"Saved static expert dataset to: {output_path}")
    print(f"states: {np.asarray(states).shape}, actions: {np.asarray(actions).shape}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate static expert data for velocity-space navigation."
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--episode-length", type=int, default=450)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--origin-x", type=float, default=316.0)
    parser.add_argument("--origin-y", type=float, default=217.0)
    parser.add_argument("--v-max-low", type=float, default=5 * math.pi / 180)
    parser.add_argument("--v-max-high", type=float, default=16 * math.pi / 180)
    parser.add_argument("--delta-v-limit", type=float, default=0.025)
    parser.add_argument("--delta-theta-limit", type=float, default=math.pi / 8)
    parser.add_argument("--k-v", type=float, default=0.02)
    parser.add_argument("--dt-scale", type=float, default=10.0)
    parser.add_argument("--distance-weight", type=float, default=0.005)
    parser.add_argument("--arrive-radius", type=float, default=5.0)
    parser.add_argument("--arrive-hold-steps", type=int, default=15)
    parser.add_argument("--log-interval", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    collect_dataset(parse_args())
