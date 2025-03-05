from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch import vmap
from torch.func import stack_module_state, functional_call
from torch import Tensor

from components.networks import MLP, CNN
from components.utils import weight_init


class DeterministicVectorActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        hidden_features = 400
        num_layers = 3
        self.net = MLP(obs_dim, act_dim, hidden_features, num_layers)

    def forward(self, obs, **kwargs):
        return F.tanh(self.net(obs))
    
    def __call__(self, obs: Tensor, **kwargs) -> Tensor:
        return self.forward(obs)


class DeterministicImageVectorActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.cnn = CNN()
        hidden_features = 400
        num_layers = 3
        in_dim = self.cnn.out_features + obs_dim
        out_dim = act_dim
        self.mlp = MLP(in_dim, out_dim, hidden_features, num_layers)

    def forward(self, img, vec, **kwargs):
        cnn_features = self.cnn(img)
        features = torch.cat([cnn_features, vec], dim=-1)
        return F.tanh(self.mlp(features))

    def __call__(self, img: Tensor, vec: Tensor, **kwargs) -> Tensor:
        return self.forward(img, vec)


class StochasticVectorActor(nn.Module):
    log_std_min = -20
    log_std_max = 2

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        hidden_features = 400
        num_layers = 3
        self.net = MLP(obs_dim, 2 * act_dim, hidden_features, num_layers)

    def forward(self, obs, use_mean=False):
        out = self.net(obs)
        mean, log_std = out.chunk(2, dim=-1)
        std = _get_std(log_std, self.log_std_min, self.log_std_max)
        dist = D.Normal(mean, std)

        if use_mean:
            act = mean
        else:
            act = dist.rsample()
        logp = torch.sum(dist.log_prob(act), dim=-1)

        squashed_act, squashed_logp = _squash(act, logp)
        return squashed_act, squashed_logp
    
    def __call__(self, obs: Tensor, use_mean: bool = False, **kwargs) -> tuple[Tensor, Tensor]:
        return self.forward(obs, use_mean=use_mean)
    

class StochasticImageVectorActor(nn.Module):
    log_std_min = -20
    log_std_max = 2

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.cnn = CNN()
        hidden_features = 400
        num_layers = 3
        in_dim = self.cnn.out_features + obs_dim
        out_dim = 2 * act_dim
        self.mlp = MLP(in_dim, out_dim, hidden_features, num_layers)

    def forward(self, img, vec, use_mean=False):
        cnn_features = self.cnn(img)
        features = torch.cat([cnn_features, vec], dim=-1)

        out = self.mlp(features)
        mean, log_std = out.chunk(2, dim=-1)
        std = _get_std(log_std, self.log_std_min, self.log_std_max)
        dist = D.Normal(mean, std)

        if use_mean:
            act = mean
        else:
            act = dist.rsample()
        logp = torch.sum(dist.log_prob(act), dim=-1)

        squashed_act, squashed_logp = _squash(act, logp)
        return squashed_act, squashed_logp
    
    def __call__(self, img: Tensor, vec: Tensor, use_mean: bool = False, **kwargs) -> tuple[Tensor, Tensor]:
        return self.forward(img, vec, use_mean=use_mean)
    

class EnsembleActor(nn.Module):
    def __init__(self, actor: nn.Module, num_ensembles: int):
        super().__init__()
        self.ml = nn.ModuleList()
        for _ in range(num_ensembles):
            copied = weight_init(deepcopy(actor))
            self.ml.append(copied)

        # vectorization
        self.base_m = deepcopy(actor).to('meta')
        def fmodel(params, buffers, *args, **kwargs):
            return functional_call(self.base_m, (params, buffers), *args, **kwargs)
        self.fmodel = fmodel

    def forward(self, *args, **kwargs):
        params, buffers = stack_module_state(self.ml)
        out_ensemble = vmap(
            self.fmodel,
            in_dims=(0, 0, None),
        )(params, buffers, *args, **kwargs)
        out_ensemble = map(
            lambda x: torch.stack(x, dim=0).mean(dim=0),
            zip(*out_ensemble),
        )
        return tuple(out_ensemble)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def _get_std(log_std: Tensor, log_std_min: float, log_std_max: float):
    log_std = torch.tanh(log_std)
    log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
    return log_std.exp()

def _squash(act: Tensor, logp: Tensor):
    squashed_act = torch.tanh(act)
    squashed_logp = logp - torch.log(1 - squashed_act ** 2 + 1e-6).sum(-1)
    return squashed_act, squashed_logp