from copy import deepcopy

import torch
import torch.nn as nn
from torch import Tensor

from components.networks import MLP, CNN
from components.utils import soft_update_params


class VectorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        hidden_features = 400
        num_layers = 3
        self.critic1 = MLP(obs_dim + act_dim, 1, hidden_features, num_layers)
        self.target1 = deepcopy(self.critic1).requires_grad_(False)
        self.critic2 = MLP(obs_dim + act_dim, 1, hidden_features, num_layers)
        self.target2 = deepcopy(self.critic2).requires_grad_(False)

    def forward(self, obs, act, target=False):
        features = torch.cat((obs, act), dim=-1)
        if not target:
            q1 = self.critic1(features)
            q2 = self.critic2(features)
        else:
            q1 = self.target1(features)
            q2 = self.target2(features)
        return q1, q2
    
    def __call__(self, obs: Tensor, act: Tensor, target: bool = False) -> tuple[Tensor, Tensor]:
        return self.forward(obs, act, target=target)
    
    def update_target(self, tau: float):
        soft_update_params(self.critic1, self.target1, tau)
        soft_update_params(self.critic2, self.target2, tau)
    

class ImageVectorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.critic1 = _SingleImageVectorCritic(obs_dim, act_dim)
        self.target1 = deepcopy(self.critic1).requires_grad_(False)
        self.critic2 = _SingleImageVectorCritic(obs_dim, act_dim)
        self.target2 = deepcopy(self.critic2).requires_grad_(False)

    def forward(self, img, vec, act, target=False):
        if not target:
            q1 = self.critic1(img, vec, act)
            q2 = self.critic2(img, vec, act)
        else:
            q1 = self.target1(img, vec, act)
            q2 = self.target2(img, vec, act)
        return q1, q2
    
    def __call__(self, img: Tensor, vec: Tensor, act: Tensor, target: bool = False) -> tuple[Tensor, Tensor]:
        return self.forward(img, vec, act, target=target)
    
    def update_target(self, tau: float):
        soft_update_params(self.critic1, self.target1, tau)
        soft_update_params(self.critic2, self.target2, tau)
    

class _SingleImageVectorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.cnn = CNN()
        hidden_features = 400
        num_layers = 3
        self.mlp = MLP(self.cnn.out_features + obs_dim + act_dim, 1, hidden_features, num_layers)

    def forward(self, img, vec, act):
        cnn_features = self.cnn(img)
        features = torch.cat([cnn_features, vec, act], dim=-1)
        return self.mlp(features)
