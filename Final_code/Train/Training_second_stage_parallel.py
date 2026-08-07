
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Algorithm.PPO_1_stage2 import PPO1
from SwarmEnv.RLEnv import CustomEnv
from matplotlib import pyplot as plt
from datetime import datetime
from Algorithm.PathFollowing import PathFollowing
import math
from multiprocessing import Process, Pipe
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from multiprocessing import freeze_support
from torch.optim.lr_scheduler import LambdaLR
import platform
import random

# 创建并行环境函数
def make_env(env_id, rank, seed=0):
    def _thunk():
        env = CustomEnv(worker_id=rank+153)
        #env.seed(seed+rank)
        return env
    return _thunk

class TrainingConfig:
    """"训练参数配置"""
    num_envs = 1
    episode_num = 2500100
    episode_length = 800
    dim_input_policy = 8
    dim_out_policy = 2 # B
    dim_input_critic = 8
    epsilon = 0.2
    action_low = np.array([-0.025, -math.pi/8])
    action_high = np.array([0.025, math.pi/8])
    state_range = np.array([[0, -2*math.pi],[16*math.pi/180, 2*math.pi]])
    list_curve = ['line', 'circle', 'sin', 'eight', 'zigzag']

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
# 主函数
def main():
    import sys
    print("Python version:")
    print(sys.version)
    SEED = 42
    #set_seed(SEED)

    # 滑动平均奖励
    def get_running_reward(reward_array: np.ndarray, window=25):
        running_reward = np.zeros_like(reward_array)
        for i in range(window - 1):
            running_reward[i] = np.mean(reward_array[:i + 1])
        for i in range(window - 1, len(reward_array)):
            running_reward[i] = np.mean(reward_array[i - window + 1:i + 1])
        return running_reward

    # 版本及硬件检查
    if (sys.version_info[0] < 3):
        raise Exception("ERROR: ML-Agents Toolkit (v0.3 onwards) requires Python 3")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化参数
    config = TrainingConfig
    envs = SubprocVecEnv([make_env(i, i) for i in range(config.num_envs)])
    PF = PathFollowing(config.num_envs)
    total_reward = np.zeros(config.episode_num)
    total_reward_flocking = np.zeros(config.episode_num)
    total_reward_dis = np.zeros(config.episode_num)
    total_reward_heading = np.zeros(config.episode_num)
    total_reward_final = np.zeros(config.episode_num)
    episode_dis = np.zeros(config.episode_num)
    time_step = 0 # 总的时间步
    current_episode = 0
    episode_reward = np.zeros([config.num_envs, config.episode_length])
    episode_reward_flocking = np.zeros([config.num_envs, config.episode_length])
    episode_reward_dis = np.zeros([config.num_envs, config.episode_length])
    episode_reward_heading = np.zeros([config.num_envs, config.episode_length])
    episode_reward_final = np.zeros([config.num_envs, config.episode_length])
    memory = [[] for _ in range(config.num_envs)]
    last_theta = np.zeros(config.num_envs)
    last_psi = np.zeros(config.num_envs)
    desire_angle = np.zeros(config.num_envs)
    robot_pos = np.ones([config.num_envs, 2])*300
    diff_ini = np.zeros(config.num_envs)
    PF.reference_points = []
    for index in range(config.num_envs):
        PF.reference_points.append(PF.generate_trajectory('line', 50))
    for index, reference_points in enumerate(PF.reference_points):
        envs.env_method("set_reference_points", reference_points, indices=[index])
    envs.env_method("set_episode_length", config.episode_length)
    cycle_step = np.zeros(config.num_envs)
    PPO = PPO1(config.dim_input_policy, config.dim_out_policy, config.dim_input_critic, config.epsilon, device, config.action_low, config.action_high, config.state_range,
                     K_epochs=5, hidden_dim=64, gamma=0.99, tau=0.01, lr_actor=1e-5, lr_critic=1e-4, Lambda=0.95)

    # 创建文件夹
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    res_dir = r".\model_data\results" + current_time
    os.makedirs(res_dir, exist_ok=True)
    loss_pi_rec = []
    
    loss_v_rec = []
   
    # 加载网络参数
    res_dir_train = r".\model_data\pretrain_static_policy_2026-05-05_11-31-34"
    #res_dir_train = r".\Algorithm"
    pretrained_params_actor = torch.load(
    os.path.join(res_dir_train, 'model376500.pt'),
    map_location=torch.device('cpu')
    )
    for actor_parameter in pretrained_params_actor:
        PPO.agent.policy.load_state_dict(actor_parameter)
        PPO.agent.target_policy.load_state_dict(actor_parameter)
    
    # 配置流速参数
    if current_episode < 800:
        flow_x = np.random.uniform(-0.2, 0.2)
    else:
        flow_x = np.random.uniform(-0.5, 0.5)
    flow_x = 15*math.pi/180 *0.7 *5*0
    envs.env_method("set_flow_x", flow_x)
    envs.env_method("set_v_max", 16*math.pi/180)

    # 学习率调度器
    lr_lambda = lambda step: 1 if step < 8000 else 0.5
    scheduler = LambdaLR(PPO.agent.policy_optimizer, lr_lambda=lr_lambda)
    y_loc = np.zeros(config.num_envs)
    change_episode = []
    obs= envs.reset()
    rec = 16
    for step in range(config.episode_length*config.episode_num):
        time_step += 1
        current_goals = []
        for index in range(config.num_envs):
            PF.t[index] += 1
        for index in range(config.num_envs):
            cycle_step[index] += 1
        for index in range(config.num_envs):
            current_goal = PF.getCurrentPoint(robot_pos[index], index)
            current_goals.append(current_goal)
        # for i, goal in enumerate(obs):
        #     envs.env_method("set_previous_obs", goal, indices=[i])  # 指定环境索引
        for i, current_goal in enumerate(current_goals):
            envs.env_method("set_current_goal", current_goal, indices=[i])
 

        # 生成所有子环境的动作
        actions = []
        theta_list = []
        psi_list = []
        for index in range(config.num_envs):
            obs_norm = obs[index, :] / state_scale
            action, theta, psi = \
                PPO.select_action(obs_norm, last_theta[index], last_psi[index])
            actions.append(action)
            theta_list.append(theta)
            psi_list.append(psi)


        # 处理动作数据格式
        action = np.array(actions)
        theta = np.array(theta_list)
        psi = np.array(psi_list)
        actions = []
        for i in range(config.num_envs):
            action_new = np.concatenate([
                np.array(action[i]),
                np.array([theta[i]]),
                np.array([psi[i]]),
                np.array([cycle_step[i]])
            ])
            actions.append(action_new)
        
        # 执行动作并记录过程数据
        next_obs, rewards, dones, infos = envs.step(actions)
        episode_reward[:, step % config.episode_length] = rewards
        robot_pos = np.array([info['pos'] for info in infos])
        done_info = np.array([info['dones'] for info in infos])
        dis = np.array([info['dis'] for info in infos])
        reward_flocking = np.array([info['reward_flocking'] for info in infos])
        reward_dis = np.array([info['reward_dis'] for info in infos])
        reward_heading = np.array([info['reward_heading'] for info in infos])
        reward_final = np.array([info['reward_final'] for info in infos])
        flow = np.array([info['flow'] for info in infos])
        flow_x_real = np.array([info['flow_x'] for info in infos])
        flow_y_real = np.array([info['flow_y'] for info in infos])
        episode_reward_flocking[:, step % config.episode_length] = reward_flocking
        episode_reward_dis[:, step % config.episode_length] = reward_dis
        episode_reward_heading[:, step % config.episode_length] = reward_heading
        episode_reward_final[:, step % config.episode_length] = reward_final
        for jdex in range(config.num_envs):
            traj_return = episode_reward[jdex, :].sum()
            if traj_return < -500:
                debug = 1
        obs_norm = obs[index] / state_scale
        next_obs_norm = next_obs[index] / state_scale
        for index in range(config.num_envs):
            memory[index].append([obs_norm, action[index][:2], rewards[index], next_obs_norm, dones[index]])
        obs = next_obs
        last_theta = theta
        last_psi = psi

        for index in range(config.num_envs):
            if cycle_step[index] == 1:
                dis11 = math.sqrt(next_obs[index, 1]*next_obs[index, 1]+next_obs[index, 0]*next_obs[index, 0])
                diff_ini[index] = dis11
        
        # 更新策略网络
        if len(PPO.memory) >= 102400:
            loss_policy, loss_value = PPO.update()
            loss_v_rec.append(loss_value)
            loss_pi_rec.append(loss_policy)
            memory = [[] for _ in range(config.num_envs)]
        
        # 检查当前episode是否结束
        for index in range(len(done_info)):
            done = done_info[index]
            if done:
                adv = 0
                temp = []
                memory_current = memory[index]
                for item in reversed(memory_current):
                    state = torch.from_numpy(item[0]).to(torch.float)
                    value_s = PPO.agent.critic(state.to(device)).item()
                    next_state = torch.from_numpy(item[3]).to(torch.float)
                    value_s_ = PPO.agent.critic(next_state.to(device)).item()
                    td_error = item[2] + PPO.gamma * value_s_ * (1 - item[4]) - value_s
                    adv = adv * PPO.gamma * PPO.Lambda + td_error
                    temp.append([item[0], item[1], item[2], item[3], item[4], [adv]])
                for item in reversed(temp):
                    PPO.push_data(item)
                # 计算每个环境在本episode的累计奖励
                env_cumulative_rewards = episode_reward[index, :].sum()  # 形状：(config.num_envs,)
                env_cumulative_rewards_flocking = episode_reward_flocking[index, :].sum()  # 形状：(config.num_envs,)
                env_cumulative_rewards_dis = episode_reward_dis[index, :].sum()  # 形状：(config.num_envs,)
                env_cumulative_rewards_heading = episode_reward_heading[index, :].sum()  # 形状：(config.num_envs,)
                env_cumulative_rewards_final = episode_reward_final[index, :].sum()  # 形状：(config.num_envs,)
                episode_reward[index, :] = np.zeros([1, config.episode_length])
                episode_reward_flocking[index, :] = np.zeros([1, config.episode_length])
                episode_reward_dis[index, :] = np.zeros([1, config.episode_length])
                episode_reward_heading[index, :] = np.zeros([1, config.episode_length])
                episode_reward_final[index, :] = np.zeros([1, config.episode_length])
                # 记录平均累计奖励（可选）
                if env_cumulative_rewards < -500:
                    env_cumulative_rewards = -500
                total_reward[current_episode] = env_cumulative_rewards
                total_reward_flocking[current_episode] = env_cumulative_rewards_flocking
                total_reward_dis[current_episode] = env_cumulative_rewards_dis
                total_reward_heading[current_episode] = env_cumulative_rewards_heading
                total_reward_final[current_episode] = env_cumulative_rewards_final
                episode_dis[current_episode] = dis[index]
                # 打印每个环境的奖励（调试用）
                # print(f"Episode {current_episode} - Env rewards: {env_cumulative_rewards} reward_flocking: {env_cumulative_rewards_flocking} reward_dis: { env_cumulative_rewards_dis} "
                #       f"reward_heading: {env_cumulative_rewards_heading} reward_final: {env_cumulative_rewards_final} End_step: {cycle_step[index]}"
                #       f"initial_diff: {diff_ini[index]} final_diff: {dis[index]} flow:{flow[index]} flow_x:{flow_x_real[index]} flow_y:{flow_y_real[index]} y_loc:{y_loc[index]}")
                current_episode += 1
                # 初始化当前值
                reference_points = PF.generate_trajectory('circle', 50)
                PF.reference_points[index] = reference_points
                PF.current[index] = 0
                envs.env_method("set_reference_points", reference_points, indices=[index])
                PF.t[index] = 0
                memory[index] = []
                cycle_step[index] = 0
                random_v_max = np.random.uniform(5*math.pi/180, 16*math.pi/180)
                v_flow =  random_v_max*0.7*5*1 *0
                envs.env_method("set_v_max", random_v_max, indices=[index])
                envs.env_method("set_flow_x", v_flow, indices=[index])
                #obs[index, :] = envs.env_method("reset", indices=[index])[0]
                obs= envs.reset()
                last_theta[index] = 0
                last_psi[index] = 0
                desire_angle[index] = 0
                symbol = True
                scheduler.step()  # 关键作用在这里！
                print(f"Episode {current_episode}, LR: {scheduler.get_last_lr()[0]}, v_max: {rec}")
                rec = random_v_max *180/math.pi
            # 保存过程数据
            if current_episode % 1500 == 0 and current_episode != 0 and symbol:
                torch.save([PPO.agent.policy.state_dict()], os.path.join(res_dir, 'model' + str(current_episode) + '.pt'))
                torch.save([PPO.agent.critic.state_dict()],
                           os.path.join(res_dir, 'model_critic' + str(current_episode) + '.pt'))
                np.save(os.path.join(res_dir, 'rewards.npy'), total_reward)
                fig, ax = plt.subplots()
                x = range(1, current_episode + 1)
                # ax.plot(x, total_reward[:current_episode], label='All')
                # ax.plot(x, total_reward_flocking[:current_episode], label='flocking')
                # ax.plot(x, total_reward_dis[:current_episode], label='dis')
                #ax.plot(x, total_reward_heading[:current_episode], label='heading')
                # ax.plot(x, total_reward_final[:current_episode], label='final')
                # 滑动平均
                ax.plot(x, get_running_reward(total_reward[:current_episode]), label="All_avr")
                ax.legend()
                ax.set_xlabel('episode')
                ax.set_ylabel('reward')
                title = f'Reward_over_episode{current_episode}'
                ax.set_title(title)
                for ep in change_episode:
                    ax.axvline(x=ep, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
                plt.savefig(os.path.join(res_dir, title))
                fig1, ax = plt.subplots()
                loss_pi_rec1 = [item for sublist in loss_pi_rec for item in sublist]
                training_epoch = len(loss_pi_rec1)
                x = range(1, training_epoch)
                ax.plot(x, loss_pi_rec1[:-1], label='policy_loss')
                ax.legend()
                ax.set_xlabel('epoch')
                ax.set_ylabel('loss')
                title = f'Policy-loss{current_episode}'
                ax.set_title(title)
                plt.savefig(os.path.join(res_dir, title))

                fig2, ax = plt.subplots()
                loss_v_rec1 = [item for sublist in loss_v_rec for item in sublist]
                training_epoch = len(loss_pi_rec1)
                x = range(1, training_epoch)
                ax.plot(x, loss_v_rec1[:-1], label='value_loss')
                ax.legend()
                ax.set_xlabel('epoch')
                ax.set_ylabel('loss')
                title = f'Value_loss{current_episode}'
                ax.set_title(title)
                plt.savefig(os.path.join(res_dir, title))
                #current_episode += 1
                symbol = False

if __name__ == '__main__':
    freeze_support()  # Windows 需要此行
    main()  # 调用主函数

