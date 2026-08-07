import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, action_low, action_high):
        super(PolicyNetwork, self).__init__()
        self.net = torch.nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
        )
        self.mu = nn.Linear(128, action_dim)
        self.sigma = nn.Linear(128, action_dim)
        self.action_low = torch.tensor(action_low, dtype=torch.float32).to(device='cpu')
        self.action_high = torch.tensor(action_high, dtype=torch.float32).to(device='cpu')
        # Apply Xavier initialization
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if m is self.mu or m is self.sigma:  # 输出层特殊处理
                nn.init.uniform_(m.weight, -1e-1, 1e-1)  # 更小的初始范围
            else:
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.net(x)
        # 计算均值
        mu_raw = torch.tanh(self.mu(x))
        mu = self.action_low+(mu_raw+1.0)/2 * (self.action_high-self.action_low)
        # 计算方差
        sigma_phi_raw, sigma_psi_raw = torch.sigmoid(self.sigma(x)).split(1, dim=-1)
        sigma_phi = sigma_phi_raw * 0.015 + 0.001  # 缩放到 (0.2, 0.5)
        sigma_psi = sigma_psi_raw * 0.25 + 0.001  # 缩放到 (0.001, 0.051)
        sigma = torch.cat([sigma_phi, sigma_psi], dim=-1)
        return mu, sigma

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        # Apply Xavier initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        value = self.net(x)
        return value

def check_for_nan(tensor):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise ValueError("Tensor contains NaN values or INF values")