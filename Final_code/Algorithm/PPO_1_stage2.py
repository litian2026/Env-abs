import numpy as np
from sympy.abc import theta

from .utils.Agent_stage2 import PPOAgent
import torch
from torch import tensor,nn
from torch.distributions import Normal
import torch.nn.functional as F
import math
import random
class PPO1:
    def __init__(self, dim_input_policy, dim_out_policy, dim_input_critic, epsilon, device, action_low, action_high, state_range,
                 K_epochs=3, hidden_dim=64, gamma=0.99, tau=0.01, lr_actor=5e-5, lr_critic=5e-5, Lambda=0.9):

        self.epsilon = epsilon
        self.device = device
        self.gamma = gamma
        self.Lambda = Lambda
        self.tau = tau
        self.lr_actor = lr_actor
        self.dim_input_policy = dim_input_policy
        self.dim_out_policy = dim_out_policy
        self.lr_critic = lr_critic
        self.k_epochs = K_epochs
        self.agent = PPOAgent(dim_input_policy=dim_input_policy,
                              dim_out_policy=dim_out_policy,
                              dim_input_critic=dim_input_critic,
                              lr_actor=lr_actor,
                              lr_critic=lr_critic,
                              hidden_dim=hidden_dim,
                              device=device,
                              action_low=action_low,
                              action_high=action_high,
                              epsilon=0.1)
        self.memory = []
        self.time = 0
        self.debug = False
        self.mini_batch_size = 512
        self.test = False
        self.state_range = state_range

    def select_action(self, state, phi_last, psi_last):
        """
        Select action based on the current policy
        :param state:
        :return:
        """
        state = tensor(state, dtype=torch.float)
        with torch.no_grad():
            mean, std = self.agent.policy(state.to(device=self.device))
            dist = Normal(mean, std)
            action = dist.sample()
            action = mean
            action = action.squeeze(0)
            action = action.detach().cpu().numpy()
            delta_phi = action[0]
            delta_psi = action[1]
            self.time += 1
            phi = phi_last+delta_phi
            psi = psi_last+delta_psi
            # 处理psi
            if psi > math.pi:
                psi -= 2 * math.pi
            elif psi < -math.pi:
                psi += 2 * math.pi
             #上下界约束
            phi = np.clip(phi, self.state_range[0, 0], self.state_range[1, 0])
            psi = np.clip(psi, self.state_range[0, 1], self.state_range[1, 1])
        return action,  phi, psi

    def push_data(self, transitions):
        self.memory.append(transitions)

    def update(self):
        """
        Update the policy and value network according to the experiences
        """
        full_s, full_a, full_r, full_s_, full_done, full_gae = self.sample()
        batch_size = full_s.size(0)
        mini_batch_size = self.mini_batch_size
        loss_policy = []
        loss_value = []
        kl_divergences = []  # 新增：用于记录每个mini-batch的KL散度
        entropies = []  # 新增：用于记录熵
        # Training the agent
        for index in range(self.k_epochs):
            indices = torch.randperm(batch_size)
            end = 0
            for jdex in range(0, batch_size, mini_batch_size):
                start = end
                end = start + mini_batch_size
                idx = indices[start:end]
                s = full_s[idx].to(self.device)
                a = full_a[idx].to(self.device)
                r = full_r[idx].to(self.device)
                s_ = full_s_[idx].to(self.device)
                done = full_done[idx].to(self.device)
                gae = full_gae[idx].to(self.device)
                with torch.no_grad():
                    #Calculate loss of value function
                    #temp_value = self.agent.target_critic(s_)
                    not_done = torch.logical_not(done)
                    temp_tensor = self.agent.target_critic(s_.to(self.device))
                    # temp_tensor1 = temp_tensor*(not_done)
                    td_target = r.to(self.device)+self.gamma*self.agent.target_critic(s_.to(self.device))*not_done.to(self.device)
                    #Calculate loss of policy
                    mu_old,sigma_old = self.agent.target_policy(s.to(self.device))
                    old_dis = Normal(mu_old, sigma_old)
                    log_probs_old = old_dis.log_prob(a.to(self.device))
                    A=gae.to(self.device)
                mu,sigma = self.agent.policy(s.to(self.device))
                mu_has_nan = torch.isnan(mu).any()
                sigma_has_nan = torch.isnan(sigma).any()
                if mu_has_nan or sigma_has_nan:
                    debug=1
                new_dis = torch.distributions.normal.Normal(mu, sigma)
                log_prob_new = new_dis.log_prob(a.to(self.device))
                log_prob_new = torch.sum(log_prob_new, dim=-1, keepdim=True)
                log_probs_old = torch.sum(log_probs_old, dim=-1, keepdim=True)
                # 计算新旧策略之间的KL散度
                with torch.no_grad():
                    # KL(π_old || π) = E[log π_old - log π]
                    kl_divergence = (log_probs_old - log_prob_new).mean()
                    kl_divergences.append(kl_divergence.item())
                    # 计算熵（用于监控）
                    entropy = new_dis.entropy()
                    entropy = torch.sum(entropy, dim=-1, keepdim=True).mean()
                    entropies.append(entropy.item())
                # important sampling
                ratio = torch.exp(log_prob_new - log_probs_old)
                self.agent.policy_optimizer.zero_grad()
                #A = torch.tensor(A, dtype=torch.float).unsqueeze(1)
                #A = A.unsqueeze(1)
                entropy = new_dis.entropy()  # 计算熵
                entropy = torch.sum(entropy, dim=-1, keepdim=True)  # 对多维动作求和
                L1 = ratio * A
                L2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * A
                ent_coef = 0.01
                loss_pi = -(torch.min(L1, L2) + ent_coef * entropy).mean()
                loss_policy.append(loss_pi.item())
                loss_pi.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.policy.parameters(), max_norm=0.5)
                self.agent.policy_optimizer.step()
                critic_value = self.agent.critic(s.to(self.device))
                loss_v = F.mse_loss(td_target.detach(), self.agent.critic(s.to(self.device)))
                loss_value.append(loss_v.item())
                if loss_v.item() > 20:
                    debug = 1
                self.agent.critic_optimizer.zero_grad()
                loss_v.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), max_norm=0.5)
                self.agent.critic_optimizer.step()
        # 新增：计算本轮epoch的平均KL散度和熵
        mean_kl = np.mean(kl_divergences) if kl_divergences else 0
        mean_entropy = np.mean(entropies) if entropies else 0
        # 新增：打印监控信息
        print(f"Epoch {index}: Mean KL: {mean_kl:.4f}, Mean Entropy: {mean_entropy:.4f}, "
              f"Policy Loss: {np.mean(loss_policy):.4f}, Value Loss: {np.mean(loss_value):.4f}")
        # 新增：KL散度警告机制
        if mean_kl > 0.2:
            print(f"⚠️  WARNING: High KL divergence ({mean_kl:.4f})! Consider reducing learning rate.")
        if mean_kl > 0.5:
            print(f"🚨 CRITICAL: Very high KL divergence ({mean_kl:.4f})! Training may diverge.")
        self.agent.target_policy.load_state_dict(self.agent.policy.state_dict())
        self.agent.target_critic.load_state_dict(self.agent.critic.state_dict())
        return loss_policy, loss_value

    def sample(self):
        """
        Obtain data from the experience pooling
        :return:
        """
        l_s, l_a, l_r, l_s_, l_done, l_gae = [], [], [], [], [], []
        for item in self.memory:
            s, a, r, s_, done, gae = item
            l_s.append(torch.tensor(s, dtype=torch.float))
            l_a.append(torch.tensor(a, dtype=torch.float))
            l_r.append(torch.tensor([r], dtype=torch.float))
            l_s_.append(torch.tensor(s_, dtype=torch.float))
            l_done.append(torch.tensor([done], dtype=torch.bool))
            l_gae.append(torch.tensor(gae, dtype=torch.float))
        s = torch.stack(l_s, dim=0).detach()
        a = torch.stack(l_a, dim=0).detach()
        r = torch.stack(l_r, dim=0).detach()
        s_ = torch.stack(l_s_, dim=0).detach()
        done = torch.stack(l_done, dim=0).detach()
        gae = torch.stack(l_gae, dim=0).detach()
        self.memory = []
        return s, a, r, s_, done, gae


