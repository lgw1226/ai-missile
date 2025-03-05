from typing import Tuple, Optional

import numpy as np
from numpy.typing import NDArray


class VectorReplayBuffer:
    def __init__(
            self,
            capacity: int,
            obs_dim: int,
            act_dim: int,
            seed: Optional[int] = None,
    ):
        self.capacity = capacity
        self.index = 0
        self.full = False
        self.np_random = np.random.default_rng(seed=seed)

        self.vec = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rwd = np.zeros((capacity,), dtype=np.float32)
        self.nvec = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.bool_)

    def append(self,
            obs: NDArray,
            act: NDArray,
            rwd: float,
            nobs: NDArray,
            done: bool,
    ):
        i = (self.index + 1) % self.capacity
        self.vec[i] = obs
        self.act[i] = act
        self.rwd[i] = rwd
        self.nvec[i] = nobs
        self.done[i] = done
        self.index = i
        self.full = self.full or self.index == 0

    def sample(self, batch_size: int) -> Tuple[NDArray, ...]:
        max_size = self.capacity if self.full else self.index
        idxs = self.np_random.choice(max_size, batch_size, replace=False)
        return (
            self.vec[idxs],
            self.act[idxs],
            self.rwd[idxs],
            self.nvec[idxs],
            self.done[idxs],
        )


class ImageVectorReplayBuffer:
    def __init__(
            self,
            capacity: int,
            obs_dim: int,
            act_dim: int,
            seed: Optional[int] = None,
    ):
        self.capacity = capacity
        self.index = 0
        self.full = False
        self.np_random = np.random.default_rng(seed=seed)

        c, h, w = (8, 86, 86)
        self.img = np.zeros((capacity, c, h, w), dtype=np.uint8)
        self.vec = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rwd = np.zeros((capacity,), dtype=np.float32)
        self.nimg = np.zeros((capacity, c, h, w), dtype=np.uint8)
        self.nvec = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.bool_)

    def append(self,
            obs: dict[str, NDArray],
            act: NDArray,
            rwd: float,
            nobs: dict[str, NDArray],
            done: bool,
    ):
        i = (self.index + 1) % self.capacity
        self.img[i] = obs['img']
        self.vec[i] = obs['vec']
        self.act[i] = act
        self.rwd[i] = rwd
        self.nimg[i] = nobs['img']
        self.nvec[i] = nobs['vec']
        self.done[i] = done
        self.index = i
        self.full = self.full or self.index == 0

    def sample(self, batch_size: int) -> Tuple[NDArray, ...]:
        max_size = self.capacity if self.full else self.index
        idxs = self.np_random.choice(max_size, batch_size, replace=False)
        return (
            self.img[idxs].astype(np.float32) / 255,
            self.vec[idxs],
            self.act[idxs],
            self.rwd[idxs],
            self.nimg[idxs].astype(np.float32) / 255,
            self.nvec[idxs],
            self.done[idxs],
        )


class QFILImageVectorReplayBuffer:
    def __init__(
            self,
            capacity: int,
            obs_dim: int,
            act_dim: int,
            seed: Optional[int] = None,
    ):
        self.capacity = capacity
        self.index = 0
        self.full = False
        self.np_random = np.random.default_rng(seed=seed)

        c, h, w = (8, 86, 86)
        self.img = np.zeros((capacity, c, h, w), dtype=np.uint8)
        self.vec = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rwd = np.zeros((capacity,), dtype=np.float32)
        self.nimg = np.zeros((capacity, c, h, w), dtype=np.uint8)
        self.nvec = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.exp_act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.bool_)

    def append(self,
            obs: dict[str, NDArray],
            act: NDArray,
            rwd: float,
            nobs: dict[str, NDArray],
            exp_act: NDArray,
            done: bool,
    ):
        i = (self.index + 1) % self.capacity
        self.img[i] = obs['img']
        self.vec[i] = obs['vec']
        self.act[i] = act
        self.rwd[i] = rwd
        self.nimg[i] = nobs['img']
        self.nvec[i] = nobs['vec']
        self.exp_act[i] = exp_act
        self.done[i] = done
        self.index = i
        self.full = self.full or self.index == 0

    def sample(self, batch_size: int) -> Tuple[NDArray, ...]:
        max_size = self.capacity if self.full else self.index
        idxs = self.np_random.choice(max_size, batch_size, replace=False)
        return (
            self.img[idxs].astype(np.float32) / 255,
            self.vec[idxs],
            self.act[idxs],
            self.rwd[idxs],
            self.nimg[idxs].astype(np.float32) / 255,
            self.nvec[idxs],
            self.exp_act[idxs],
            self.done[idxs],
        )
