from collections import OrderedDict

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.optim import Adam

from modules.actors import Actor
from modules.buffers import Buffer


class Dagger():

    def __init__(
            self,
            actor: Actor,
            buffer: Buffer,
            lr: float,
            device: th.device
    ):
        self.actor = actor.to(device)
        self.buffer = buffer
        self.device = device

        self.buffer.device = device
        self.optim = Adam(self.actor.parameters(), lr=lr)

    @th.no_grad()
    def get_action(self, x) -> np.ndarray:
        if type(x) is np.ndarray:
            x_t = self._t(x).unsqueeze(0)
            act = self._n(self.actor(x_t).squeeze())
        elif type(x) is OrderedDict:
            img, vec = x['img'] / 255, x['vec']
            img_t = self._t(img).unsqueeze(0)
            vec_t = self._t(vec).unsqueeze(0)
            act = self._n(self.actor(img_t, vec_t).squeeze())
        return act
    
    def update(self, batch_size):
        batch = self.buffer.sample(batch_size)
        obs = batch[:-1]
        act = batch[-1]
        pred_act = self.actor(*obs)
        loss = F.huber_loss(pred_act, act)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        log = {
            'loss/nll': loss,
            # 'train/input-mean': th.mean(obs),
            # 'train/input-std': th.std(obs),
            'train/label-mean': th.mean(act),
            'train/label-std': th.std(act),
            'train/prediction-mean': th.mean(pred_act),
            'train/prediction-std': th.std(pred_act),
        }
        return log
    
    def save(self, path: str):
        th.save(self.actor.state_dict(), path)
        
    def _t(self, a: np.ndarray):
        return th.as_tensor(a, dtype=th.float32, device=self.device)
    
    def _n(self, t: th.Tensor):
        return np.asarray(t.detach().cpu().numpy())
