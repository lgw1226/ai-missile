from typing import Tuple
from functools import partial

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.networks import MLP, ImageVectorEncoder


class Critic(nn.Module):

    def __init__(self):
        super().__init__()
        self.functions = nn.ModuleList()

    def forward(self, *args, **kwds) -> th.Tensor:
        pass


class VectorCritic(Critic):

    def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            num_critics: int = 2,
    ):
        super().__init__()
        for _ in range(num_critics):
            self.functions.append(MLP(obs_dim + act_dim, 1, 400, 3, layer_norm=True))

    def forward(self, obs, act) -> th.Tensor:
        oa = th.cat((obs, act), dim=-1)
        qs = [f(oa) for f in self.functions]
        return th.cat(qs, dim=-1)


class ConvNet(nn.Module):

    def __init__(
            self,
            img_shape: Tuple[int, ...],
            obs_dim: int,
            act_dim: int,
    ):
        super().__init__()

        hidden_dim = 256
        self.encoder = ImageVectorEncoder(img_shape, obs_dim + act_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, img, vec, act):
        img1 = img[:, 0::2]
        img2 = img[:, 1::2]
        features = self.encoder(img1, img2, th.cat((vec, act), dim=-1))
        return self.fc(features)


class ImageCritic(Critic):

    def __init__(
            self,
            img_shape: Tuple[int, ...],
            obs_dim: int,
            act_dim: int,
            num_critics: int = 2,
    ):
        super().__init__()
        for _ in range(num_critics):
            self.functions.append(ConvNet(img_shape, obs_dim, act_dim))

    def forward(self, img, vec, act) -> th.Tensor:
        qs = [f(img, vec, act) for f in self.functions]
        return th.cat(qs, dim=-1)
