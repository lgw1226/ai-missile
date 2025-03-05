import os
from itertools import count
from argparse import Namespace

import gym
import numpy as np
import torch

import wandb
import hydra
from hydra.utils import instantiate
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

from agents.qfil import VectorQFIL, ImageVectorQFIL


@hydra.main(config_path='configs', config_name='train_qfil', version_base=None)
def main(cfg: DictConfig):

    torch.manual_seed(cfg.seed)
    np_random = np.random.default_rng(seed=cfg.seed)
    device = torch.device('cuda', index=cfg.gpu_index)

    # parse environment configs
    if cfg.env.type == 'kinematics':
        if cfg.env.dim == 2:
            env_name = 'kinematics2d-v2'
            obs_dim, act_dim = 20, 1
            expert_cls = PNG_2d
        elif cfg.env.dim == 3:
            env_name = 'kinematics3d-v0'
            obs_dim, act_dim = 40, 2
            expert_cls = PNG_3d
    elif cfg.env.dim == 'dynamics':
        if cfg.env.dim == 2:
            env_name = 'dynamics2d-v0'
            obs_dim, act_dim = 20, 1
            expert_cls = SMGC_2d

    if cfg.env.obs_type == 'flame-ir':
        env_wrapper_cls = ImageStackEnvWrapper
    elif cfg.env.obs_type == 'point-ir':
        if cfg.env.type == 'kinematics':
            env_wrapper_cls = IRStackEnvWrapper
        elif cfg.env.type == 'dynamics':
            env_wrapper_cls = IRStackEnvDynWrapper
    elif cfg.env.obs_type == 'vector':
        env_wrapper_cls = VectorWrapper

    render_kwargs = {
        'env': env_name,
        'phase': 'IL',
        'algo': 'DAgger',
        'render': True,
        'os': 'ubuntu',
        'sim_idx': cfg.sim_index,
    }
    render_namespace = Namespace(**render_kwargs)

    # initialize components
    env = env_wrapper_cls(
        gym.make(env_name, **cfg.env),
        args=render_namespace
    )
    expert = expert_cls(env, gain=cfg.expert.ctrl_gain)
    agent = instantiate(
        cfg.agent.builder,
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
        _recursive_=False,
    )
    buffer = instantiate(
        cfg.buffer,
        obs_dim=obs_dim,
        act_dim=act_dim,
    )
    run = wandb.init(
        project=cfg.logging.wandb_project,
        name=cfg.logging.wandb_name,
        mode=cfg.logging.wandb_mode,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )
    save_dir = os.path.join(cfg.logging.save_dir, run.id)
    os.makedirs(save_dir, exist_ok=True)

    # DAgger phase
    for epoch in range(1, cfg.training.num_dagger_epochs + 1):
        # collect data
        beta = cfg.training.beta_decay ** epoch
        done = True
        for _ in range(cfg.training.num_dagger_samples):
            if done:
                obs, info = env.reset()
            ctrl_gain_lb = cfg.expert.ctrl_gain - cfg.expert.ctrl_gain_amp
            ctrl_gain_ub = cfg.expert.ctrl_gain + cfg.expert.ctrl_gain_amp
            expert.gain = np_random.uniform(ctrl_gain_lb, ctrl_gain_ub)
            expert_act = expert.control_input(info['Mstates'], info['Tstates'])
            mixed_act = beta * expert_act + (1 - beta) * agent.get_action(obs)
            nobs, rwd, done, info = env.step(mixed_act)
            buffer.append(obs, mixed_act, rwd, nobs, expert_act, info['success'])
            obs = nobs
        # update agent
        for epoch_step in range(1, cfg.training.num_dagger_updates + 1):
            step = (epoch - 1) * cfg.training.num_dagger_updates + epoch_step
            batch = buffer.sample(cfg.training.dagger_batch_size)
            update_log = agent.update(batch, filter=False)
            if step % cfg.logging.log_freq == 0:
                update_log.update({
                    'train/epoch': epoch,
                    'train/step': step,
                })
                run.log(update_log)
    agent.save(os.path.join(save_dir, 'dagger.pt'))
        
    # QFIL phase
    for epoch in range(epoch + 1, epoch + cfg.training.num_qfil_epochs + 1):
        # update agent
        for epoch_step in range(1, cfg.training.num_qfil_updates + 1):
            step = (epoch - 1) * cfg.training.num_qfil_updates + epoch_step
            batch = buffer.sample(cfg.training.qfil_batch_size)
            update_log = agent.update(batch, filter=True)
            if step % cfg.logging.log_freq == 0:
                update_log.update({
                    'train/epoch': epoch,
                    'train/step': step,
                })
                run.log(update_log)
    agent.save(os.path.join(save_dir, 'qfil.pt'))

    # cleanup
    env.close()


if __name__ == '__main__':
    main()