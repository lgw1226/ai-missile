from collections import OrderedDict
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer

from hydra.utils import instantiate
from omegaconf import DictConfig

from components.actors import DeterministicVectorActor, DeterministicImageVectorActor
from components.critics import VectorCritic, ImageVectorCritic


class VectorQFIL():
    def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            gamma: float,
            quantile: float,
            num_pushforward_samples: int,
            tau: float,
            actor_cfg: DictConfig,
            actor_optim_cfg: DictConfig,
            critic_cfg: DictConfig,
            critic_optim_cfg: DictConfig,
            device: torch.device = 'cpu',
    ):
        self.tau = tau
        self.gamma = gamma
        self.quantile = quantile
        self.num_pushforward_samples = num_pushforward_samples

        assert self.quantile > 0 and self.quantile < 1
        assert self.num_pushforward_samples > 0
        
        self.actor: DeterministicVectorActor = instantiate(actor_cfg, obs_dim=obs_dim, act_dim=act_dim).to(device)
        self.actor_optim: Optimizer = instantiate(actor_optim_cfg, params=self.actor.parameters())
        self.critic: VectorCritic = instantiate(critic_cfg, obs_dim=obs_dim, act_dim=act_dim).to(device)
        self.critic_optim: Optimizer = instantiate(critic_optim_cfg, params=self.critic.parameters())
        self.device = device

    @torch.no_grad()
    def get_action(self, obs: NDArray) -> NDArray:
        act = self.actor(self._tensor(obs).unsqueeze(0))
        return self._ndarray(act.squeeze())
    
    def update(self, batch: tuple[NDArray, ...], filter: bool = False) -> dict[str, float]:
        obs, act, rwd, nobs, done = map(self._tensor, batch)
        rwd = rwd.unsqueeze(-1)
        done = done.unsqueeze(-1)
        update_log = {}

        # update critic
        with torch.no_grad():
            nact = self.actor(nobs)
            nq1, nq2 = self.critic(nobs, nact, target=True)
            nq = torch.min(nq1, nq2)
            td_target = rwd + self.gamma * (1 - done) * nq
        q1, q2 = self.critic(obs, act)
        update_log.update({'train/q-mean': torch.mean(torch.min(q1, q2)).item()})
        critic_loss = (F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)) / 2

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        self.critic.update_target(self.tau)
        update_log.update({'loss/critic': critic_loss.item()})

        # quantile filtering
        if filter:
            with torch.no_grad():
                obs_ = obs.unsqueeze(1).repeat(1, self.num_pushforward_samples, 1)
                act_ = self.actor(obs_)

                q1_, q2_ = self.critic(obs_, act_, target=True)
                q_ = torch.min(q1_, q2_)
                q_quantile = torch.quantile(q_, self.quantile, dim=1)

                q = torch.min(q1, q2)
                mask = q >= q_quantile

            if torch.any(mask):
                obs = obs[mask]
                act = act[mask]

            update_log.update({
                'train/qfil-batch-ratio': torch.mean(mask, dtype=torch.float32).item(),
                'train/filtered-q-mean': torch.mean(q[mask]).item(),
            })

        # update actor
        pred_act = self.actor(obs)
        actor_loss = F.huber_loss(pred_act, act)

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        update_log.update({'loss/actor': actor_loss.item()})

        return update_log
    
    def save_ckpt(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
        }, path)

    def load_ckpt(self, path: str):
        data = torch.load(path)
        self.actor.load_state_dict(data['actor'])
        self.actor_optim.load_state_dict(data['actor_optim'])
        self.critic.load_state_dict(data['critic'])
        self.critic_optim.load_state_dict(data['critic_optim'])
        
    def _tensor(self, a: NDArray):
        return torch.as_tensor(a, dtype=torch.float32, device=self.device)
    
    def _ndarray(self, t: Tensor):
        return np.asarray(t.detach().cpu().numpy())


class ImageVectorQFIL():
    def __init__(
            self,
            obs_dim: int,
            act_dim: int,
            gamma: float,
            quantile: float,
            num_pushforward_samples: int,
            num_critic_updates: int,
            tau: float,
            actor_cfg: DictConfig,
            actor_optim_cfg: DictConfig,
            critic_cfg: DictConfig,
            critic_optim_cfg: DictConfig,
            device: torch.device = 'cpu',
    ):
        self.gamma = gamma
        self.quantile = quantile
        self.num_pushforward_samples = num_pushforward_samples
        self.num_critic_updates = num_critic_updates
        self.tau = tau

        assert self.quantile > 0 and self.quantile < 1
        assert self.num_pushforward_samples > 0
        
        self.actor: DeterministicImageVectorActor = instantiate(actor_cfg, obs_dim=obs_dim, act_dim=act_dim).to(device)
        self.actor_optim: Optimizer = instantiate(actor_optim_cfg, params=self.actor.parameters())
        self.critic: ImageVectorCritic = instantiate(critic_cfg, obs_dim=obs_dim, act_dim=act_dim).to(device)
        self.critic_optim: Optimizer = instantiate(critic_optim_cfg, params=self.critic.parameters())
        self.device = device

    @torch.no_grad()
    def get_action(self, obs: dict[str, NDArray]) -> NDArray:
        act = self.actor(self._tensor(obs['img']).unsqueeze(0), self._tensor(obs['vec']).unsqueeze(0))
        return self._ndarray(act.squeeze())
    
    def update(self, batch: tuple[NDArray, ...], filter: bool = False) -> dict[str, float]:
        img, vec, act, rwd, nimg, nvec, exp_act, done = map(self._tensor, batch)
        rwd = rwd.unsqueeze(-1)
        done = done.unsqueeze(-1)
        update_log = {}

        # update critic
        for _ in range(self.num_critic_updates):
            with torch.no_grad():
                nact = self.actor(nimg, nvec)
                nq1, nq2 = self.critic(nimg, nvec, nact, target=True)
                nq = torch.min(nq1, nq2)
                td_target = rwd + self.gamma * (1 - done) * nq
            q1, q2 = self.critic(img, vec, act)
            update_log.update({'train/q-mean': torch.mean(torch.min(q1, q2)).item()})
            critic_loss = (F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)) / 2

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
            self.critic.update_target(self.tau)
            update_log.update({'loss/critic': critic_loss.item()})

        # quantile filtering
        if filter:
            with torch.no_grad():
                img_ = img.unsqueeze(1).repeat(1, self.num_pushforward_samples, 1, 1, 1)
                vec_ = vec.unsqueeze(1).repeat(1, self.num_pushforward_samples, 1)
                act_ = self.actor(img_, vec_)

                q1_, q2_ = self.critic(img_, vec_, act_, target=True)
                q_ = torch.min(q1_, q2_)
                q_quantile = torch.quantile(q_, self.quantile, dim=1)

                exp_q1, exp_q2 = self.critic(img, vec, exp_act, target=True)
                exp_q = torch.min(exp_q1, exp_q2)
                mask = exp_q >= q_quantile

            if torch.any(mask):
                img = img[mask]
                vec = vec[mask]
                exp_act = exp_act[mask]

            update_log.update({
                'train/qfil-batch-ratio': torch.mean(mask, dtype=torch.float32).item(),
                'train/filtered-q-mean': torch.mean(exp_q[mask]).item(),
            })

        # update actor
        pred_act = self.actor(img, vec)
        actor_loss = F.huber_loss(pred_act, exp_act)

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        update_log.update({'loss/actor': actor_loss.item()})

        return update_log
    
    def save_ckpt(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
        }, path)

    def load_ckpt(self, path: str):
        data = torch.load(path)
        self.actor.load_state_dict(data['actor'])
        self.actor_optim.load_state_dict(data['actor_optim'])
        self.critic.load_state_dict(data['critic'])
        self.critic_optim.load_state_dict(data['critic_optim'])
        
    def _tensor(self, a: NDArray):
        return torch.as_tensor(a, dtype=torch.float32, device=self.device)
    
    def _ndarray(self, t: Tensor):
        return np.asarray(t.detach().cpu().numpy())