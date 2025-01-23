import os
from functools import partial
from argparse import Namespace

import gym
import numpy as np
import torch as th

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from gym_Missile.control_methods.PNG_2d import PNG_2d
from gym_Missile.control_methods.PNG_3d import PNG_3d
from gym_Missile.control_methods.SMGC_2d import SMGC_2d
from gym_Missile.render.EnvWrapper import (
    ImageStackEnvWrapper,
    IRStackEnvWrapper,
    IRStackEnvDynWrapper,
    VectorWrapper,
)

from agents.dagger import Dagger
from modules.actors import (
    DeterministicImageActor,
    DeterministicVectorActor,
    EnsembleActor,
)
from modules.buffers import (
    ImageBuffer,
    VectorBuffer,
)


@hydra.main(config_path='configs', config_name='train_dagger', version_base=None)
def main(cfg: DictConfig):

    # unpack arguments
    gpu_index = cfg.gpu_index
    sim_index = cfg.sim_index
    
    num_updates = cfg.num_updates
    num_rollouts = cfg.num_rollouts
    num_steps = cfg.num_steps

    batch_size = cfg.batch_size
    n_eval = cfg.n_eval
    eval_freq = cfg.eval_freq
    log_freq = cfg.log_freq

    env_type = cfg.environment.env_type
    env_dim = cfg.environment.env_dim
    obs_type = cfg.environment.obs_type
    ctrl_gain = cfg.environment.ctrl_gain
    beta_decay = cfg.environment.beta_decay

    if env_type == 'kinematics':
        if env_dim == 2:
            env_name = 'kinematics2d-v1'
            max_episode_length = 2000
            obs_dim, act_dim = 20, 1
            ctrl_cls = PNG_2d
        elif env_dim == 3:
            env_name = 'kinematics3d-v0'
            max_episode_length = 5000
            obs_dim, act_dim = 40, 2
            ctrl_cls = PNG_3d
    elif env_type == 'dynamics':
        if env_dim == 2:
            env_name = 'dynamics2d-v0'
            max_episode_length = 3000
            obs_dim, act_dim = 20, 1
            ctrl_cls = SMGC_2d

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
    ctrl = ctrl_cls(env, gain=ctrl_gain)

    # create agent
    device = th.device('cuda', index=gpu_index)
    if obs_type in ['target-ir', 'point-ir']:
        img_shape = (8, 86, 86)
        actor = EnsembleActor(
            DeterministicImageActor,
            (img_shape, obs_dim, act_dim), {}, cfg.agent.num_actors,
        )
        buffer = ImageBuffer(cfg.agent.buffer_size, img_shape, obs_dim, act_dim)
    elif obs_type == 'vector':
        actor = EnsembleActor(
            DeterministicVectorActor,
            (obs_dim, act_dim), {}, cfg.agent.num_actors,
        )
        buffer = VectorBuffer(cfg.agent.buffer_size, obs_dim, act_dim)
    agent = Dagger(actor, buffer, cfg.agent.lr, device)

    def evaluate():
        ep_rwd = 0
        ep_len = 0
        min_dist = 0
        success = 0
        for _ in range(n_eval):
            obs, info = env.reset()
            for t in range(max_episode_length):
                act = agent.get_action(obs)
                obs, rwd, done, info = env.step(act)
                ep_rwd += rwd
                if done: break
            ep_len += t + 1
            min_dist += info['min_dist']
            success += info['success']
        eval_log = {
            'eval/success': success / n_eval,
            'eval/return': ep_rwd / n_eval,
            'eval/episode-length': ep_len / n_eval,
            'eval/minimum-distance': min_dist / n_eval,
        }
        return eval_log
            
    run = wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        group=cfg.wandb.group,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )

    total_steps = 0
    num_iters = 0
    while total_steps < num_updates:
        
        beta = beta_decay ** num_iters
        for _ in range(num_rollouts):
            obs, info = env.reset()
            for _ in range(max_episode_length):
                ctrl_act = ctrl.control_input(info['Mstates'], info['Tstates'])
                act = beta * ctrl_act + (1 - beta) * agent.get_action(obs)
                obs, _, done, info = env.step(act)
                agent.buffer.add(obs, ctrl_act)
                if done: break
        num_iters += 1

        for _ in range(num_steps):
            log = agent.update(batch_size)
            total_steps += 1
            if total_steps % eval_freq == 0:
                eval_log = evaluate()
                log.update(eval_log)
            if total_steps % log_freq == 0:
                log.update({'train/step': total_steps})
                run.log(log)
                print(log)
            if total_steps == num_updates: break

    env.close()
    os.makedirs('trained_models/', exist_ok=True)
    agent.save(f'trained_models/{run.id}.pt')


if __name__ == '__main__':
    main()