from functools import partial
from typing import Tuple, Dict, Any
from abc import *
from math import floor

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from modules.networks import ImageVectorEncoder


class Actor(nn.Module):

    is_stochastic = True


class DeterministicVectorActor(Actor):
    '''Return bounded and deterministic actions given observations.'''

    is_stochastic = False

    def __init__(
            self,
            obs_dim: int,
            act_dim: int,
    ):
        super().__init__()

        hidden_dim = 400
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, act_dim)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, obs):
        features = F.relu(self.ln(self.fc1(obs)))
        features = F.relu(self.ln(self.fc2(features)))
        return F.tanh(self.fc3(features))
    

class SquashedNormal(D.Normal):

    def __init__(self, loc: th.Tensor, scale: th.Tensor):
        super().__init__(loc, scale)
    
    def rsample(self):
        normal_rsample = super().rsample()
        return th.tanh(normal_rsample)

    def log_prob(self, value):
        # normal_value = th.atanh(th.clip(value, -1 + 1e-6, 1 - 1e-6))
        normal_value = th.atanh(value)
        normal_logp = super().log_prob(normal_value)
        logp = th.sum(normal_logp - th.log(1 - th.square(value) + 1e-6), dim=-1, keepdim=True)
        return logp


class StochasticVectorActor(Actor):

    def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            logstd_bound: Tuple[float, float],
    ):
        super().__init__()
        self.logstd_bound = logstd_bound

        hidden_dim = 400
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)

        self.mean = nn.Linear(hidden_dim, act_dim)
        self.logstd = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs, use_mean=False):
        features = F.relu(self.ln(self.fc1(obs)))
        features = F.relu(self.ln(self.fc2(features)))

        mean = self.mean(features)
        lb, ub = self.logstd_bound
        logstd = 0.5 * ((ub - lb) * th.tanh(self.logstd(features)) + (ub + lb))
        std = th.exp(logstd)
        dist = D.Normal(mean, std)

        unbounded_act = mean if use_mean else dist.rsample()
        unbounded_logp = dist.log_prob(unbounded_act)

        act = th.tanh(unbounded_act)
        logp = th.sum(unbounded_logp - th.log(1 - th.square(act) + 1e-6), dim=-1, keepdim=True)
        return act, logp
    

class StochasticImageActor(Actor):

    def __init__(
            self,
            img_shape: Tuple[int, ...],
            obs_dim: int,
            act_dim: int,
            logstd_bound: Tuple[float, float],
    ):
        super().__init__()
        self.logstd_bound = logstd_bound

        hidden_dim = 256
        self.encoder = ImageVectorEncoder(img_shape, obs_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, act_dim)
        self.logstd = nn.Linear(hidden_dim, act_dim)

    def forward(self, img, vec, use_mean=False):
        img1 = img[:, 0::2]
        img2 = img[:, 1::2]
        features = self.encoder(img1, img2, vec)

        mean = self.mean(features)
        lb, ub = self.logstd_bound
        logstd = 0.5 * ((ub - lb) * th.tanh(self.logstd(features)) + (ub + lb))
        std = th.exp(logstd)
        dist = D.Normal(mean, std)

        unbounded_act = mean if use_mean else dist.rsample()
        unbounded_logp = dist.log_prob(unbounded_act)

        act = th.tanh(unbounded_act)
        logp = th.sum(unbounded_logp - th.log(1 - th.square(act) + 1e-6), dim=-1, keepdim=True)
        return act, logp
    

class DeterministicImageActor(Actor):
    '''Return bounded and deterministic actions given image observations.'''

    is_stochastic = False

    def __init__(
            self,
            img_shape: Tuple[int, ...],
            obs_dim: int,
            act_dim: int,
    ):
        super().__init__()

        hidden_dim = 256
        self.encoder = ImageVectorEncoder(img_shape, obs_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, act_dim)

    def forward(self, img, vec):
        img1 = img[:, 0::2]
        img2 = img[:, 1::2]
        features = self.encoder(img1, img2, vec)
        return self.fc(features)
    

class EnsembleActor(Actor):
    '''Aggregate multiple actors into a single ensemble actor.'''

    def __init__(
            self,
            actor_cls: Actor,
            actor_args: Tuple[Any, ...],
            actor_kwargs: Dict[str, Any],
            num_actors: int,
    ):
        assert (
            actor_cls == DeterministicImageActor or
            actor_cls == DeterministicVectorActor
        )
        super().__init__()
        self.is_stochastic = actor_cls.is_stochastic

        self.actors = nn.ModuleList()
        actor_init = partial(actor_cls, *actor_args, **actor_kwargs)
        for _ in range(num_actors): self.actors.append(actor_init())

    def forward(self, *args):
        acts = th.stack([actor(*args) for actor in self.actors], dim=-1)
        act = th.mean(acts, dim=-1)
        return act
