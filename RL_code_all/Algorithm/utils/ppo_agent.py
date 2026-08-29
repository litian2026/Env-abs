from copy import deepcopy

from torch import Tensor
import torch
import numpy as np
from torch.optim import Adam
import torch.nn.functional as F

from .networks import PolicyNetwork, ValueNetwork, HistoryPolicyNetwork, HistoryValueNetwork

class PPOAgent(object):
    """
    General class for ppo agents (policy, critic, target_policy, target_critic...)
    """
    def __init__(self,dim_input_policy,dim_out_policy,dim_input_critic,lr_actor,lr_critic,hidden_dim,device,action_low, action_high, epsilon=0.1, policy_kind="mlp", current_state_dim=None, history_dim=0, history_latent_dim=2, history_window=15, history_feature_dim=8):
        """
        Initialize the PPOAgent with the given parameters
        :param dim_input_policy:
        :param dim_out_policy:

        :param dim_input_critic:
        :param lr_actor:
        :param lr_critic:
        :param hidden_dim:
        :param device:
        :param epsilon:
        """
        if policy_kind == "history":
            if current_state_dim is None or history_dim <= 0:
                raise ValueError("history policy requires current_state_dim and history_dim")
            self.policy = HistoryPolicyNetwork(current_state_dim, history_dim, dim_out_policy, action_low, action_high, history_latent_dim, history_window, history_feature_dim).to(device)
            self.critic = HistoryValueNetwork(current_state_dim, history_dim, history_latent_dim, history_window, history_feature_dim).to(device)
        else:
            self.policy = PolicyNetwork(dim_input_policy,dim_out_policy,action_low,action_high).to(device)
            self.critic = ValueNetwork(dim_input_critic).to(device)
        self.policy_optimizer = Adam(self.policy.parameters(),lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(),lr_critic)

        self.target_policy = deepcopy(self.policy).to(device)
        self.target_critic = deepcopy(self.critic).to(device)

        self.epsilon = epsilon
        self.device = device


    def action(self,obs,*,model_out=False):

        logits = self.policy(obs)
        action = self.gumble_softmax(logits)
        action = torch.tanh(logits)
        if model_out:
            return action, logits
        return action

    def target_action(self,obs):
        logits = self.target_policy(obs)
        action = self.gumble_softmax(logits)
        action = torch.tanh(logits)


    def critic_value(self, state_list:Tensor,act_list:Tensor):
        x = torch.cat((state_list,act_list),1)
        return self.critic(x).squeeze(1)

    def target_critic_value(self,state_list:Tensor,act_list:Tensor):
        x  = torch.cat((state_list,act_list))
        return self.critic(x).squeeze(1)


    @staticmethod
    def gumble_softmax(logits,tau=1.0,eps=1e-20):
        epsilon = torch.rand_like(logits)
        logits += -torch.log(-torch.log(epsilon + eps) + eps)
        return F.softmax(logits / tau, dim=-1)











