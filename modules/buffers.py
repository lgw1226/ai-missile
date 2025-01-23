from typing import Tuple, Dict

import torch as th
import numpy as np


np.ndarray = np.ndarray
th.Tensor = th.Tensor


class Buffer():

    device = None

    def add(self): raise NotImplementedError

    def sample(self, batch_size: int) -> Tuple[th.Tensor, ...]: raise NotImplementedError


class ImageBuffer(Buffer):
    '''Store image observations of given shape along with corresponding actions.'''

    def __init__(
            self,
            size: int,
            img_shape: Tuple[int, ...],
            obs_dim: int,
            act_dim: int,
    ):
        self.size = size
        c, h, w = img_shape

        self.img_buf = np.zeros((size, c, h, w), dtype=np.uint8)
        self.vec_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(self, obs: Dict[str, np.ndarray], act: np.ndarray):
        img = obs['img']
        vec = obs['vec']

        self.img_buf[self.pos] = img
        self.vec_buf[self.pos] = vec
        self.act_buf[self.pos] = act

        self.pos += 1
        if self.pos == self.size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> Tuple[th.Tensor, ...]:
        buf_len = self.pos if not self.full else self.size
        idxs = np.random.randint(0, buf_len, size=(batch_size,))

        img, vec, act = tuple(map(self._t, (
            self.img_buf[idxs] / 255,
            self.vec_buf[idxs],
            self.act_buf[idxs],
        )))

        return img, vec, act
    
    def _t(self, a: np.ndarray) -> th.Tensor:
        return th.as_tensor(a, dtype=th.float32, device=self.device)
    

class EstimatorBuffer(Buffer):
    '''Store image observations of given shape along with corresponding actions.'''

    def __init__(
            self,
            size: int,
            img_shape: Tuple[int, ...],
            obs_dim: int,
            label_dim: int,
            device: th.device = th.device('cpu'),
    ):
        self.size = size
        self.device = device
        c, h, w = img_shape

        self.img1_buf = np.zeros((size, c, h, w), dtype=np.uint8)
        self.img2_buf = np.zeros((size, c, h, w), dtype=np.uint8)
        self.vec_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.label_buf = np.zeros((size, label_dim), dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(self, obs: Dict[str, np.ndarray], act: np.ndarray):
        img = obs['img']
        img1 = img[0::2]
        img2 = img[1::2]
        vec = obs['vec']

        self.img1_buf[self.pos] = img1
        self.img2_buf[self.pos] = img2
        self.vec_buf[self.pos] = vec
        self.label_buf[self.pos] = act

        self.pos += 1
        if self.pos == self.size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> Tuple[th.Tensor, ...]:
        buf_len = self.pos if not self.full else self.size
        idxs = np.random.randint(0, buf_len, size=(batch_size,))

        return tuple(map(self._t, (
            self.img1_buf[idxs] / 255,
            self.img2_buf[idxs] / 255,
            self.vec_buf[idxs],
            self.label_buf[idxs],
        )))
    
    def _t(self, a: np.ndarray) -> th.Tensor:
        return th.as_tensor(a, dtype=th.float32, device=self.device)
    

class VectorBuffer(Buffer):
    '''Store vector observations along with actions.'''

    def __init__(
            self,
            size: int,
            obs_dim: int,
            act_dim: int,
    ):
        self.size = size

        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(self, obs: np.ndarray, act: np.ndarray):
        self.obs_buf[self.pos] = obs
        self.act_buf[self.pos] = act

        self.pos += 1
        if self.pos == self.size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> Tuple[th.Tensor, ...]:
        buf_len = self.pos if not self.full else self.size
        idxs = np.random.randint(0, buf_len, size=(batch_size,))

        return tuple(map(self._t, (self.obs_buf[idxs], self.act_buf[idxs])))
    
    def _t(self, a: np.ndarray) -> th.Tensor:
        return th.as_tensor(a, dtype=th.float32, device=self.device)


class VectorReplayBuffer(Buffer):

    def __init__(
            self,
            size: int,
            obs_dim: int,
            act_dim: int,
    ):
        self.size = size

        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rwd_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(
            self,
            obs: np.ndarray,
            act: np.ndarray,
            rwd: np.ndarray,
            next_obs: np.ndarray,
            done: np.ndarray,
    ):
        self.obs_buf[self.pos] = obs
        self.act_buf[self.pos] = act
        self.rwd_buf[self.pos] = rwd
        self.next_obs_buf[self.pos] = next_obs
        self.done_buf[self.pos] = done

        self.pos += 1
        if self.pos == self.size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> Tuple[th.Tensor, ...]:
        buf_len = self.pos if not self.full else self.size
        idxs = np.random.randint(0, buf_len, size=(batch_size,))

        batch = (
            self.obs_buf[idxs],
            self.act_buf[idxs],
            self.rwd_buf[idxs],
            self.next_obs_buf[idxs],
            self.done_buf[idxs],
        )
        return tuple(map(self._t, batch))
    
    def _t(self, a: np.ndarray) -> th.Tensor:
        return th.as_tensor(a, dtype=th.float32, device=self.device)
    

class ImageReplayBuffer(Buffer):

    def __init__(
            self,
            size: int,
            img_shape: Tuple[int],
            vec_dim: int,
            act_dim: int,
    ):
        self.size = size

        self.img_buf = np.zeros((size + 1, *img_shape), dtype=np.uint8)
        self.vec_buf = np.zeros((size + 1, vec_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rwd_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(
            self,
            img: np.ndarray,
            vec: np.ndarray,
            act: np.ndarray,
            rwd: np.ndarray,
            next_img: np.ndarray,
            next_vec: np.ndarray,
            done: np.ndarray,
    ):
        self.img_buf[self.pos] = img
        self.vec_buf[self.pos] = vec
        self.act_buf[self.pos] = act
        self.rwd_buf[self.pos] = rwd
        self.done_buf[self.pos] = done

        self.img_buf[self.pos + 1] = next_img
        self.vec_buf[self.pos + 1] = next_vec

        self.pos += 1
        if self.pos == self.size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> Tuple[th.Tensor, ...]:
        buf_len = self.pos if not self.full else self.size
        idxs = np.random.randint(0, buf_len, size=(batch_size,))

        batch = (
            self.img_buf[idxs] / 255,
            self.vec_buf[idxs],
            self.act_buf[idxs],
            self.rwd_buf[idxs],
            self.img_buf[idxs + 1] / 255,
            self.vec_buf[idxs + 1],
            self.done_buf[idxs],
        )
        return tuple(map(self._t, batch))
    
    def _t(self, a: np.ndarray) -> th.Tensor:
        return th.as_tensor(a, dtype=th.float32, device=self.device)

def test_image_buffer():
    buffer_size = 100
    img_shape = (8, 86, 86)
    act_dim = 2
    buffer = ImageBuffer(buffer_size, img_shape, act_dim, device=th.device('cuda'))

    for _ in range(buffer_size*2):
        img = np.random.randint(0, 255, size=img_shape)
        act = np.random.randn(act_dim)
        buffer.add(img, act)
    
    batch_size = 128
    img, act = buffer.sample(batch_size)
    print(img.shape, img.dtype, th.mean(img), th.std(img))
    print(act.shape, img.dtype)

def test_vector_buffer():
    buffer_size = 100
    obs_dim = 4
    act_dim = 2
    buffer = VectorBuffer(buffer_size, obs_dim, act_dim, device=th.device('cuda'))

    for _ in range(buffer_size*2):
        obs = np.random.randn(obs_dim)
        act = np.random.randn(act_dim)
        buffer.add(obs, act)
    
    batch_size = 128
    obs, act = buffer.sample(batch_size)
    print(obs.shape, obs.dtype)
    print(act.shape, act.dtype)


if __name__ == '__main__':

    test_image_vector_buffer()
