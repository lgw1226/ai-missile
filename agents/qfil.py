from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.optim import Adam

from modules.actors import Actor
from modules.critics import Critic
from modules.buffers import Buffer


class QFIL():

    def __init__(
            self,
            actor: Actor,
            critic: Critic,
            buffer: Buffer,
            lr: float,
            device: th.device
    ):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.critic_target = deepcopy(self.critic).requires_grad_(False)
        self.buffer = buffer
        self.device = device

        self.buffer.device = device
        self.optims = {
            'actor': Adam(self.actor.parameters(), lr=lr),
            'critic': Adam(self.critic.parameters(), lr=lr),
        }

    @th.no_grad()
    def get_action(self, x, use_mean=True) -> np.ndarray:
        if type(x) is np.ndarray:
            x_t = self._t(x).unsqueeze(0)
            act, _ = self.actor(x_t, use_mean=True)
            act = self._n(act.squeeze())
        elif type(x) is OrderedDict:
            img, vec = x['img'] / 255, x['vec']
            img_t = self._t(img).unsqueeze(0)
            vec_t = self._t(vec).unsqueeze(0)
            act, _ = self.actor(img_t, vec_t, use_mean=True)
            act = self._n(act.squeeze())
        return act
    
    def update(self, batch_size: int, quantile: float = 0):
        log = {}
        obs, act, rwd, next_obs, done = self.buffer.sample(batch_size)

        with th.no_grad():
            next_act, _ = self.actor(next_obs)
            next_q, _ = th.min(self.critic_target(next_obs, next_act), dim=-1, keepdim=True)
            q_target = rwd + 0.99 * (1 - done) * next_q
        qs = self.critic(obs, act)
        critic_loss = th.mean(th.square(qs - q_target))
        log.update({'train/q-mean': th.mean(qs)})

        if quantile != 0:
            with th.no_grad():
                pushforward_samples = 50
                obs_ = obs.unsqueeze(1).repeat(1, pushforward_samples, 1)
                act_, _ = self.actor(obs_)
                q_, _ = th.min(self.critic(obs_, act_), dim=-1)
                q_quantile = th.quantile(q_, quantile, dim=-1)
                q, _ = th.min(qs.detach(), dim=-1)
                mask = q >= q_quantile
            if mask.any():
                obs, act = tuple(map(lambda t: t[mask], (obs, act)))
                log.update({
                    'train/filtered-q-mean': th.mean(q[mask]),
                    'train/q-quantile': th.mean(q_quantile),
                })
        pred_act, pred_logp = self.actor(obs)
        actor_loss = F.huber_loss(pred_act, act)

        loss = critic_loss + actor_loss
        for optim in self.optims.values(): optim.zero_grad()
        loss.backward()
        for optim in self.optims.values(): optim.step()
        self.update_target(0.995)

        log.update({
            'loss/actor': actor_loss,
            'loss/critic': critic_loss,
            'train/label-mean': th.mean(act),
            'train/prediction-mean': th.mean(pred_act),
            'train/prediction-logp': th.mean(pred_logp),
        })
        return log
    
    def update_target(self, tau: float):
        for f, ft in zip(self.critic.functions, self.critic_target.functions):
            sd = f.state_dict()
            sdt = ft.state_dict()
            for key in sd:
                sdt[key] = (1 - tau) * sd[key] + tau * sdt[key]
            ft.load_state_dict(sdt)
    
    def save(self, path: str):
        th.save(self.actor.state_dict(), path)
        
    def _t(self, a: np.ndarray):
        return th.as_tensor(a, dtype=th.float32, device=self.device)
    
    def _n(self, t: th.Tensor):
        return np.asarray(t.detach().cpu().numpy())


class ImageQFIL():

    def __init__(
            self,
            actor: Actor,
            critic: Critic,
            buffer: Buffer,
            lr: float,
            device: th.device,
    ):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.critic_target = deepcopy(self.critic).requires_grad_(False)
        self.buffer = buffer
        self.device = device

        self.buffer.device = device
        self.optims = {
            'actor': Adam(self.actor.parameters(), lr=lr),
            'critic': Adam(self.critic.parameters(), lr=lr),
        }

    @th.no_grad()
    def get_action(self, x, use_mean=False) -> np.ndarray:
        if type(x) is np.ndarray:
            x_t = self._t(x).unsqueeze(0)
            act, _ = self.actor(x_t, use_mean=use_mean)
            act = self._n(act.squeeze())
        elif type(x) is OrderedDict:
            img, vec = x['img'] / 255, x['vec']
            img_t = self._t(img).unsqueeze(0)
            vec_t = self._t(vec).unsqueeze(0)
            act, _ = self.actor(img_t, vec_t, use_mean=use_mean)
            act = self._n(act.squeeze())
        return act
    
    def update(self, batch_size: int, quantile: float = 0):
        log = {}
        img, vec, act, rwd, next_img, next_vec, done = self.buffer.sample(batch_size)

        with th.no_grad():
            next_act, _ = self.actor(next_img, next_vec)
            next_q, _ = th.min(self.critic_target(next_img, next_vec, next_act), dim=-1, keepdim=True)
            q_target = rwd + 0.99 * (1 - done) * next_q
        qs = self.critic(img, vec, act)
        critic_loss = th.mean(th.square(qs - q_target))
        log.update({'train/q-mean': th.mean(qs)})

        if quantile != 0:
            with th.no_grad():
                pushforward_samples = 20
                img_ = img.repeat_interleave(pushforward_samples, dim=0)
                vec_ = vec.repeat_interleave(pushforward_samples, dim=0)
                act_, _ = self.actor(img_, vec_)
                q_, _ = th.min(self.critic(img_, vec_, act_), dim=-1)
                q_quantile = th.quantile(q_.reshape(-1, pushforward_samples), quantile, dim=-1)
                q, _ = th.min(qs.detach(), dim=-1)
                mask = q >= q_quantile
            if mask.any():
                img, vec, act = tuple(map(lambda t: t[mask], (img, vec, act)))
                log.update({
                    'train/qfil-batch-ratio': th.mean(mask, dtype=th.float32),
                    'train/filtered-q-mean': th.mean(q[mask]),
                    'train/q-quantile': th.mean(q_quantile[mask]),
                })
        pred_act, pred_logp = self.actor(img, vec)
        actor_loss = F.huber_loss(pred_act, act)

        loss = critic_loss + actor_loss
        for optim in self.optims.values(): optim.zero_grad()
        loss.backward()
        for optim in self.optims.values(): optim.step()
        self.update_target(0.995)

        log.update({
            'loss/actor': actor_loss,
            'loss/critic': critic_loss,
            'train/label-mean': th.mean(act),
            'train/prediction-mean': th.mean(pred_act),
            'train/prediction-logp': th.mean(pred_logp),
        })
        return log
    
    def update_target(self, tau: float):
        for f, ft in zip(self.critic.functions, self.critic_target.functions):
            sd = f.state_dict()
            sdt = ft.state_dict()
            for key in sd:
                sdt[key] = (1 - tau) * sd[key] + tau * sdt[key]
            ft.load_state_dict(sdt)
    
    def save(self, path: str):
        th.save(self.actor.state_dict(), path)
        
    def _t(self, a: np.ndarray):
        return th.as_tensor(a, dtype=th.float32, device=self.device)
    
    def _n(self, t: th.Tensor):
        return np.asarray(t.detach().cpu().numpy())