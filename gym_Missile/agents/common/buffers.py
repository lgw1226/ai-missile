import numpy as np
import torch
import matplotlib.pyplot as plt
from gym_Missile.agents.common.data_augs import *
import time
class ReplayBuffer(object):
    """
    A simple FIFO experience replay buffer for agents.
    """

    def __init__(self, obs_dim, act_dim, size, device):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def add(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample(self, batch_size=64):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=torch.Tensor(self.obs1_buf[idxs]).to(self.device),
                    obs2=torch.Tensor(self.obs2_buf[idxs]).to(self.device),
                    acts=torch.Tensor(self.acts_buf[idxs]).to(self.device),
                    rews=torch.Tensor(self.rews_buf[idxs]).to(self.device),
                    done=torch.Tensor(self.done_buf[idxs]).to(self.device))

class ImageReplayBuffer(object):
    """
    A simple FIFO experience replay buffer for agents.
    """

    def __init__(self, image_obs_dim, act_dim, size, device):
        self.obs1_buf = np.zeros([size] + image_obs_dim, dtype=np.float32)
        self.obs2_buf = np.zeros([size] + image_obs_dim, dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.capacity = size
        self.idx = 0
        self.full = False
        self.device = device

    def add(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.idx] = obs
        self.obs2_buf[self.idx] = next_obs
        self.acts_buf[self.idx] = act
        self.rews_buf[self.idx] = rew
        self.done_buf[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size=64, augmentation = False):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )
        obs1 = self.obs1_buf[idxs] #numpy 128 6 256 256
        obs2 = self.obs2_buf[idxs]
        if augmentation:
            obs1 = random_cutout(obs1)
            obs1 = random_jiterring(obs1)
            obs2 = random_cutout(obs2)
            obs2 = random_jiterring(obs2)
            ##
            #image_telephoto = ((obs1[0][0]*255).astype(np.uint8))
            #image_telephoto = np.repeat(np.expand_dims(image_telephoto, axis=2), 3, axis=2)
            #plt.imshow(image_telephoto)
            #plt.show()
            ##
        return dict(obs1=torch.Tensor(obs1).to(self.device),
                    obs2=torch.Tensor(obs2).to(self.device),
                    acts=torch.Tensor(self.acts_buf[idxs]).to(self.device),
                    rews=torch.Tensor(self.rews_buf[idxs]).to(self.device),
                    done=torch.Tensor(self.done_buf[idxs]).to(self.device))

class StateReplayBuffer(object):
    """
    A simple FIFO experience replay buffer for agents.
    """

    def __init__(self, size):
        self.Mstate_buf = np.zeros([size, 9], dtype=np.float32)
        self.Tstate_buf = np.zeros([size, 9], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def add(self, Mstate, Tstate):
        self.Mstate_buf[self.ptr] = Mstate
        self.Tstate_buf[self.ptr] = Tstate
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def state_sample(self, batch_size=1):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return self.Mstate_buf[idxs],self.Tstate_buf[idxs]



class Buffer(object):
    """
    A buffer for storing trajectories experienced by a agent interacting
    with the environment.
    """

    def __init__(self, obs_dim, act_dim, size, device, gamma=0.99, lam=0.97):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.act_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.don_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.v_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.max_size = 0, size
        self.device = device

    def add(self, obs, act, rew, don, v):
        assert self.ptr < self.max_size      # Buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.don_buf[self.ptr] = don
        self.v_buf[self.ptr] = v
        self.ptr += 1

    def finish_path(self):
        previous_v = 0
        running_ret = 0
        running_adv = 0
        for t in reversed(range(len(self.rew_buf))):
            # The next two line computes rewards-to-go, to be targets for the value function
            running_ret = self.rew_buf[t] + self.gamma*(1-self.don_buf[t])*running_ret
            self.ret_buf[t] = running_ret

            # The next four lines implement GAE-Lambda advantage calculation
            running_del = self.rew_buf[t] + self.gamma*(1-self.don_buf[t])*previous_v - self.v_buf[t]
            running_adv = running_del + self.gamma*self.lam*(1-self.don_buf[t])*running_adv
            previous_v = self.v_buf[t]
            self.adv_buf[t] = running_adv
        # The next line implement the advantage normalization trick
        self.adv_buf = (self.adv_buf - self.adv_buf.mean()) / self.adv_buf.std()
        
    def get(self):
        assert self.ptr == self.max_size     # Buffer has to be full before you can get
        self.ptr = 0
        return dict(obs=torch.Tensor(self.obs_buf).to(self.device),
                    act=torch.Tensor(self.act_buf).to(self.device),
                    ret=torch.Tensor(self.ret_buf).to(self.device),
                    adv=torch.Tensor(self.adv_buf).to(self.device),
                    v=torch.Tensor(self.v_buf).to(self.device))
        