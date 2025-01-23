import os
from itertools import count
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

from agents.qfil import QFIL, ImageQFIL
from modules.actors import (
    DeterministicImageActor,
    DeterministicVectorActor,
    StochasticVectorActor,
    StochasticImageActor,
    EnsembleActor,
)
from modules.critics import (
    VectorCritic,
    ImageCritic,
)
from modules.buffers import (
    VectorBuffer,
    ImageBuffer,
    VectorReplayBuffer,
    ImageReplayBuffer
)


@hydra.main(config_path='configs', config_name='train_qfil', version_base=None)
def main(cfg: DictConfig):

    # unpack arguments
    gpu_index = cfg.gpu_index
    sim_index = cfg.sim_index

    num_dagger_updates = cfg.num_dagger_updates
    num_dagger_rollouts = cfg.num_dagger_rollouts
    num_dagger_steps = cfg.num_dagger_steps
    num_qfil_updates = cfg.num_qfil_updates
    qfil_quantile = cfg.qfil_quantile

    batch_size = cfg.batch_size
    n_eval = cfg.n_eval
    eval_freq = cfg.eval_freq
    log_freq = cfg.log_freq

    env_type = cfg.environment.env_type
    env_dim = cfg.environment.env_dim
    obs_type = cfg.environment.obs_type
    ctrl_gain = cfg.environment.ctrl_gain
    ctrl_gain_amp = cfg.environment.ctrl_gain_amp
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
            max_episode_length = 2000
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
        actor = StochasticImageActor(img_shape, obs_dim, act_dim, (-5, 0))
        critic = ImageCritic(img_shape, obs_dim, act_dim)
        buffer = ImageReplayBuffer(cfg.agent.buffer_size, img_shape, obs_dim, act_dim)
        agent = ImageQFIL(actor, critic, buffer, cfg.agent.lr, device)
    elif obs_type == 'vector':
        actor = StochasticVectorActor(obs_dim, act_dim, (-5, 0))
        critic = VectorCritic(obs_dim, act_dim)
        buffer = VectorReplayBuffer(cfg.agent.buffer_size, obs_dim, act_dim)
        agent = QFIL(actor, critic, buffer, cfg.agent.lr, device)


    def evaluate():
        ep_rwd = 0
        ep_len = 0
        min_dist = 0
        success = 0
        for _ in range(n_eval):
            obs, info = env.reset()
            for t in range(max_episode_length):
                act = agent.get_action(obs, use_mean=True)
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

    # Dagger phase
    total_steps = 0
    num_iters = 0
    while total_steps < num_dagger_updates:

        beta = beta_decay ** num_iters
        for _ in range(num_dagger_rollouts):
            obs, info = env.reset()
            for t in range(max_episode_length):
                ctrl.gain = np.random.uniform(ctrl_gain - ctrl_gain_amp, ctrl_gain + ctrl_gain_amp)
                ctrl_act = ctrl.control_input(info['Mstates'], info['Tstates'])
                act = beta * ctrl_act + (1 - beta) * agent.get_action(obs)
                next_obs, rwd, done, info = env.step(act)
                if obs_type in ['point-ir', 'target-ir']:
                    agent.buffer.add(obs['img'], obs['vec'], ctrl_act, rwd, next_obs['img'], next_obs['vec'], done)
                elif obs_type == 'vector':
                    agent.buffer.add(obs, ctrl_act, rwd, next_obs, done)
                obs = next_obs
                if done: break

        for _ in range(num_dagger_steps):
            log = agent.update(batch_size)
            total_steps += 1
            if total_steps % eval_freq == 0:
                eval_log = evaluate()
                log.update(eval_log)
            if total_steps % log_freq == 0:
                log.update({'train/step': total_steps})
                run.log(log)
                print(log)
        
    # QFIL phase
    for _ in range(num_qfil_updates):
        log = agent.update(batch_size * 2, quantile=qfil_quantile)
        total_steps += 1
        if total_steps % eval_freq == 0:
            eval_log = evaluate()
            log.update(eval_log)
        if total_steps % log_freq == 0:
            log.update({'train/step': total_steps})
            run.log(log)
            print(log)

    os.makedirs('trained_models/', exist_ok=True)
    agent.save(f'trained_models/{run.id}.pt')
    env.close()


if __name__ == '__main__':
    main()