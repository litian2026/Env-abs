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
        self.register_buffer('action_low', torch.tensor(action_low, dtype=torch.float32), persistent=False)
        self.register_buffer('action_high', torch.tensor(action_high, dtype=torch.float32), persistent=False)
        # Apply Xavier initialization
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if m is self.mu or m is self.sigma:  
                nn.init.uniform_(m.weight, -1e-1, 1e-1)  
            else:
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.net(x)
        mu_raw = torch.tanh(self.mu(x))
        mu = self.action_low+(mu_raw+1.0)/2 * (self.action_high-self.action_low)
        sigma_phi_raw, sigma_psi_raw = torch.sigmoid(self.sigma(x)).split(1, dim=-1)
        sigma_phi = sigma_phi_raw * 0.02 + 0.001  
        sigma_psi = sigma_psi_raw * 0.2 + 0.001  
        sigma = torch.cat([sigma_phi, sigma_psi], dim=-1)
        return mu, sigma

class HistoryPolicyNetwork(nn.Module):
    def __init__(self, current_state_dim, history_dim, action_dim, action_low, action_high,
                 latent_dim=2, history_window=15, history_feature_dim=8, gru_hidden_dim=64):
        super(HistoryPolicyNetwork, self).__init__()
        self.current_state_dim = current_state_dim
        self.history_dim = history_dim
        self.history_window = history_window
        self.history_feature_dim = history_feature_dim
        if history_window * history_feature_dim != history_dim:
            raise ValueError("history_dim must equal history_window * history_feature_dim")
        self.gru = nn.GRU(
            input_size=history_feature_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True,
        )
        self.to_latent = nn.Sequential(
            nn.Linear(gru_hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, latent_dim),
            nn.Tanh(),
        )
        self.policy = PolicyNetwork(current_state_dim + latent_dim, action_dim, action_low, action_high)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        current_state = x[:, :self.current_state_dim]
        history = x[:, self.current_state_dim:]
        history = history.reshape(-1, self.history_window, self.history_feature_dim)
        _, hidden = self.gru(history)
        latent = self.to_latent(hidden[-1])
        return self.policy(torch.cat([current_state, latent], dim=-1))


class HistoryValueNetwork(nn.Module):
    def __init__(self, current_state_dim, history_dim, latent_dim=2,
                 history_window=15, history_feature_dim=8, gru_hidden_dim=64):
        super(HistoryValueNetwork, self).__init__()
        self.current_state_dim = current_state_dim
        self.history_dim = history_dim
        self.history_window = history_window
        self.history_feature_dim = history_feature_dim
        if history_window * history_feature_dim != history_dim:
            raise ValueError("history_dim must equal history_window * history_feature_dim")
        self.gru = nn.GRU(
            input_size=history_feature_dim,
            hidden_size=gru_hidden_dim,
            batch_first=True,
        )
        self.to_latent = nn.Sequential(
            nn.Linear(gru_hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, latent_dim),
            nn.Tanh(),
        )
        self.value = ValueNetwork(current_state_dim + latent_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        current_state = x[:, :self.current_state_dim]
        history = x[:, self.current_state_dim:]
        history = history.reshape(-1, self.history_window, self.history_feature_dim)
        _, hidden = self.gru(history)
        latent = self.to_latent(hidden[-1])
        return self.value(torch.cat([current_state, latent], dim=-1))


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