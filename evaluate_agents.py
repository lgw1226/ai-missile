import os
from functools import partial
from abc import *
from typing import Dict, Any
from argparse import Namespace

import gym
import numpy as np
import torch as th

import hydra
import pandas

import gym_Missile
from gym_Missile.render.EnvWrapper import (
    ImageStackEnvWrapper,
    IRStackEnvWrapper,
    IRStackEnvDynWrapper,
    VectorWrapper,
)
from gym_Missile.control_methods.PNG_2d import PNG_2d
from gym_Missile.control_methods.PNG_3d import PNG_3d
from gym_Missile.control_methods.SMGC_2d import SMGC_2d

from modules.actors import (
    DeterministicImageActor,
    DeterministicVectorActor,
    EnsembleActor,
    StochasticVectorActor,
    StochasticImageActor,
)

class Agent(metaclass=ABCMeta):

    device: th.device = None

    @abstractmethod
    def get_action(self, obs: np.ndarray, info: Dict[str, Any]) -> np.ndarray: pass

    def _t(self, a: np.ndarray) -> th.Tensor:
        return th.as_tensor(a, dtype=th.float32, device=self.device)
    
    def _n(self, t: th.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()


class PNG_2D(Agent):

    def __init__(
            self,
            png: PNG_2d,
            *args,
            **kwargs,
    ):
        self.png = png
    
    def get_action(self, obs, info):
        act = self.png.control_input(info['Mstates'], info['Tstates'])
        return act


class PNG_3D(Agent):

    def __init__(
            self,
            png: PNG_3d,
            *args,
            **kwargs,
    ):
        self.png = png
    
    def get_action(self, obs, info):
        act = self.png.control_input(info['Mstates'], info['Tstates'])
        return act


class SMGC_2D(Agent):

    def __init__(
            self,
            smgc: SMGC_2d,
            *args,
            **kwargs,
    ):
        self.smgc = smgc
    
    def get_action(self, obs, info):
        act = self.smgc.control_input(info['Mstates'], info['Tstates'])
        return act


class RandomPNG(Agent):

    def __init__(
            self,
            png: PNG_2d,
    ):
        self.png = png
    
    def get_action(self, obs, info):
        self.png.N_PN = np.random.uniform(3, 7)
        act = self.png.control_input(info['Mstates'], info['Tstates'])
        return act
    

class VectorDagger(Agent):
    
    def __init__(
            self,
            path: str,
            obs_dim: int,
            act_dim: int,
            num_actors: int = 4,
            device: th.device = None,
    ):
        self.device = device
        self.actor = EnsembleActor(
            DeterministicVectorActor,
            (obs_dim, act_dim), {}, num_actors,
        ).to(device).eval()
        self.actor.load_state_dict(th.load(path, map_location=device))

    @th.no_grad()
    def get_action(self, obs, info):
        obs_t = self._t(obs)
        act_t = self.actor(obs_t)
        return self._n(act_t)
    

class ImageDagger(Agent):
    
    def __init__(
            self,
            path: str,
            obs_dim: int,
            act_dim: int,
            num_actors: int = 4,
            device: th.device = None,
    ):
        self.device = device
        self.actor = EnsembleActor(
            DeterministicImageActor,
            ((8, 86, 86), obs_dim, act_dim), {}, num_actors,
        ).to(device).eval()
        self.actor.load_state_dict(th.load(path, map_location=device))

    @th.no_grad()
    def get_action(self, obs, info):
        img = self._t(obs['img'] / 255).unsqueeze(0)
        vec = self._t(obs['vec']).unsqueeze(0)
        act_t = self.actor(img, vec)
        return self._n(act_t.squeeze())
    
    
class VectorQFIL(Agent):
    
    def __init__(
            self,
            path: str,
            obs_dim: int,
            act_dim: int,
            device: th.device = None,
    ):
        self.device = device
        self.actor = StochasticVectorActor(obs_dim, act_dim, (-5, 0)).to(device).eval()
        self.actor.load_state_dict(th.load(path, map_location=device))

    @th.no_grad()
    def get_action(self, obs, info):
        obs_t = self._t(obs)
        act_t, _ = self.actor(obs_t, use_mean=True)
        return self._n(act_t)


class ImageQFIL(Agent):
    
    def __init__(
            self,
            path: str,
            obs_dim: int,
            act_dim: int,
            device: th.device = None,
    ):
        self.device = device
        self.actor = StochasticImageActor((8, 86, 86), obs_dim, act_dim, (-5, 0)).to(device).eval()
        self.actor.load_state_dict(th.load(path, map_location=device))

    @th.no_grad()
    def get_action(self, obs, info):
        img = self._t(obs['img'] / 255).unsqueeze(0)
        vec = self._t(obs['vec']).unsqueeze(0)
        act_t, _ = self.actor(img, vec, use_mean=True)
        return self._n(act_t.squeeze())


def evaluate_2d(env: gym.Env, agent: Agent, options: Dict[str, float]):
    ep_rwd = 0
    ep_act = 0
    options['success_dist'] = 0
    obs, info = env.reset(options=options)
    for t in range(env.spec.max_episode_steps):
        act = agent.get_action(obs, info)
        obs, rwd, done, info = env.step(act)
        ep_rwd += rwd
        ep_act += np.abs(act)
        if done: break
    ep_len = t + 1
    min_dist = info['min_dist']
    eval_result = (
        options['tau'],
        options['theta_m'],
        options['theta_t'],
        float(ep_act / ep_len),
        ep_rwd,
        ep_len,
        min_dist,
    )
    print(eval_result)
    return eval_result

def evaluate_2d_dynamics(env: gym.Env, agent: Agent, options: Dict[str, float]):
    ep_rwd = 0
    ep_act = 0
    options['success_dist'] = 0
    obs, info = env.reset(options=options)
    for t in range(env.spec.max_episode_steps):
        act = agent.get_action(obs, info)
        obs, rwd, done, info = env.step(act)
        ep_rwd += rwd
        ep_act += np.abs(act)
        if done: break
    ep_len = t + 1
    min_dist = info['min_dist']
    eval_result = (
        options['theta_m'],
        options['theta_t'],
        float(ep_act / ep_len),
        ep_rwd,
        ep_len,
        min_dist,
    )
    print(eval_result)
    return eval_result

def evaluate_3d(env: gym.Env, agent: Agent, options: Dict[str, float]):
    ep_rwd = 0
    ep_act = 0
    options['success_dist'] = 0
    obs, info = env.reset(options=options)
    for t in range(env.spec.max_episode_steps):
        act = agent.get_action(obs, info)
        obs, rwd, done, info = env.step(act)
        ep_rwd += rwd
        ep_act += np.abs(act)
        if done: break
    ep_len = t + 1
    min_dist = info['min_dist']
    eval_result = (
        options['tau'],
        options['T_pitch'],
        options['T_yaw'],
        float(ep_act[0] / ep_len),
        float(ep_act[1] / ep_len),
        ep_rwd,
        ep_len,
        min_dist,
    )
    print(eval_result)
    return eval_result


@hydra.main(config_path='configs', config_name='evaluate_agents', version_base=None)
def main(cfg):

    # environment
    # unpack arguments
    gpu_index = cfg.gpu_index
    sim_index = cfg.sim_index

    env_type = cfg.environment.env_type
    env_dim = cfg.environment.env_dim
    obs_type = cfg.environment.obs_type

    if env_type == 'kinematics':
        if env_dim == 2:
            env_name = 'kinematics2d-v1'
            max_episode_length = 2000
            obs_dim, act_dim = 20, 1
        elif env_dim == 3:
            env_name = 'kinematics3d-v0'
            max_episode_length = 5000
            obs_dim, act_dim = 40, 2
    elif env_type == 'dynamics':
        if env_dim == 2:
            env_name = 'dynamics2d-v0'
            max_episode_length = 3000
            obs_dim, act_dim = 20, 1

    if obs_type == 'target-ir':
        wrapper = ImageStackEnvWrapper
    elif obs_type == 'point-ir':
        if env_type == 'kinematics':
            wrapper = IRStackEnvWrapper
        elif env_type == 'dynamics':
            wrapper = IRStackEnvDynWrapper
    elif obs_type == 'vector':
        wrapper = VectorWrapper

    render_kwargs = {
        'env': env_name,
        'phase': 'IL',
        'algo': 'DAgger',
        'render': True,
        'os': 'ubuntu',
        'sim_idx': sim_index,
    }
    render_namespace = Namespace(**render_kwargs)

    env = wrapper(gym.make(env_name), args=render_namespace)
    env.spec.max_episode_steps = max_episode_length

    # agent
    # unpack arguments
    agent_type = cfg.agent.type
    agent_dir = cfg.agent.dir
    agent_fname = cfg.agent.fname

    agent_path = os.path.join(agent_dir, agent_fname + '.pt')
    csv_filename = '-'.join([agent_type, obs_type, agent_fname])
    num_actors = 4
    device = th.device('cuda', index=gpu_index)
    agent_args = (agent_path, obs_dim, act_dim)
    agent_kwds = {'device': device}

    if agent_type == 'qfil' and obs_type in ['point-ir', 'target-ir']:
        agent_cls = ImageQFIL
    elif agent_type == 'qfil' and obs_type == 'vector':
        agent_cls = VectorQFIL
    elif agent_type == 'dagger' and obs_type in ['point-ir', 'target-ir']:
        agent_cls = ImageDagger
        agent_kwds['num_actors'] = num_actors
    elif agent_type == 'dagger' and obs_type == 'vector':
        agent_cls = VectorDagger
        agent_kwds['num_actors'] = num_actors
    elif agent_type == 'png':
        if env_dim == 2:
            png = PNG_2d(env)
            agent_cls = partial(PNG_2D, png)
        elif env_dim == 3:
            png = PNG_3d(env)
            agent_cls = partial(PNG_3D, png)
    elif agent_type == 'smgc':
        if env_dim == 2:
            smgc = SMGC_2d(env)
            agent_cls = partial(SMGC_2D, smgc)
    agent = agent_cls(*agent_args, **agent_kwds)

    # evaluate
    os.makedirs('eval_results/', exist_ok=True)
    options = {}

    if env_dim == 2 and env_type == 'dynamics':
        data = []
        for theta_m in np.arange(0, 40 + 1, 5):
            options['theta_m'] = theta_m
            for theta_t in np.arange(0, 180 + 1, 5):
                options['theta_t'] = theta_t
                data.append(evaluate_2d_dynamics(env, agent, options))
        cols = ['theta_m', 'theta_t', 'episode_action', 'episode_return', 'episode_length', 'minimum_distance']
        dataframe = pandas.DataFrame(
            data=data,
            columns=cols,
            dtype=np.float32,
        )
        dataframe.to_csv(f'eval_results/{csv_filename}.csv')

    elif env_dim == 2 and env_type == 'kinematics':
        data = []
        for tau in [0.1, 0.2, 0.3]:
            options = {'tau': tau}
            for theta_m in np.arange(0, 40 + 1, 5):
                options['theta_m'] = theta_m
                for theta_t in np.arange(0, 180 + 1, 5):
                    options['theta_t'] = theta_t
                    data.append(evaluate_2d(env, agent, options))
        cols = ['tau', 'theta_m', 'theta_t', 'episode_action', 'episode_return', 'episode_length', 'minimum_distance']
        dataframe = pandas.DataFrame(
            data=data,
            columns=cols,
            dtype=np.float32,
        )
        dataframe.to_csv(f'eval_results/{csv_filename}.csv')

    elif env_dim == 3 and env_type == 'kinematics':
        data = []
        for tau in [0.1, 0.2, 0.3]:
            options = {'tau': tau}
            for pitch_t in (-70 + np.arange(20) * 3):
                options['T_pitch'] = pitch_t
                for yaw_t in (-165 + np.arange(20) * 3):
                    options['T_yaw'] = yaw_t
                    data.append(evaluate_3d(env, agent, options))
            for pitch_t in (10 + np.arange(15) * 4):
                options['T_pitch'] = pitch_t
                for yaw_t in (15 + np.arange(15) * 4):
                    options['T_yaw'] = yaw_t
                    data.append(evaluate_3d(env, agent, options))
        cols = ['tau', 'T_pitch', 'T_yaw', 'action_0', 'action_1', 'episode_return', 'episode_length', 'minimum_distance']
        dataframe = pandas.DataFrame(
            data=data,
            columns=cols,
            dtype=np.float32,
        )
        dataframe.to_csv(f'eval_results/{csv_filename}.csv')


if __name__ == '__main__': main()
