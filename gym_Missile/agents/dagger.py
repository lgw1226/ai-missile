import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from gym_Missile.agents.common.utils import *
from gym_Missile.agents.common.buffers import *
from gym_Missile.agents.common.networks import *
from math import pi, atan2, cos, sin
from numpy import linalg as LA

D2R = pi / 180

class Agent(object):
   """An implementation of the Deep Deterministic Policy Gradient (DDPG) agent."""

   def __init__(self,
                env,
                args,
                device,
                act_noise=0.1,
                hidden_sizes=(128,128),
                policy_lr=3e-4,
                qf_lr=3e-4,
                gradient_clip_policy=1.0,
                gradient_clip_qf=1.0,
   ):

      self.env = env
      self.args = args
      self.device = device
      self.obs_dim = int(env.obs_dim)
      self.act_dim = env.act_dim
      self.act_limit = 1
      self.act_noise = act_noise
      self.hidden_sizes = hidden_sizes
      self.policy_lr = policy_lr
      self.qf_lr = qf_lr
      self.gradient_clip_policy = gradient_clip_policy
      self.gradient_clip_qf = gradient_clip_qf

      # Main network
      self.policy = MLP(self.obs_dim, self.act_dim, self.act_limit, 
                                    hidden_sizes=self.hidden_sizes, 
                                    output_activation=torch.tanh,
                                    use_actor=True).to(self.device)
      # Create optimizers
      self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr)
      self.loss = nn.MSELoss()
      
   def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
      if not os.path.exists('checkpoints/'):
         os.makedirs('checkpoints/')
      if ckpt_path is None:
         ckpt_path = "checkpoints/dagger_checkpoint_{}_{}".format(env_name, suffix)
      print('Saving models to {}'.format(ckpt_path))
      torch.save({'policy_state_dict': self.policy.state_dict(),
                 'policy_optimizer_state_dict': self.policy_optimizer.state_dict()}, ckpt_path)

   # Load model parameters
   def load_checkpoint(self, ckpt_path, evaluate=False):
      print('Loading models from {}'.format(ckpt_path))
      if ckpt_path is not None:
         checkpoint = torch.load(ckpt_path)
         # checkpoint = torch.load(ckpt_path, map_location='cuda:0')
         self.policy.load_state_dict(checkpoint['policy_state_dict'])
         self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])

         if evaluate:
            self.policy.eval()
         else:
            self.policy.train()

   def select_action(self, state, evaluate=False):
      if not torch.is_tensor(state):
         state = torch.from_numpy(state).to(self.device)
      NN_input = torch.unsqueeze(state, dim=0)
      action = self.policy(NN_input.to(self.device)).detach().cpu().numpy()
      if(action.ndim == 2):
         action = np.squeeze(action, axis=0)
      if not evaluate:
         action += self.act_noise * np.random.randn(self.act_dim)
      return np.clip(action, -self.act_limit, self.act_limit)

   def train_model(self, memory, batch_size, updates):
      batch = memory.sample(batch_size)
      obs1 = batch['obs1']
      acts = batch['acts']

      pred_action = self.policy(obs1)
      # DAgger losses
      policy_loss = self.loss(acts, pred_action)

      # Update policy network parameter
      self.policy_optimizer.zero_grad()
      policy_loss.backward()
      self.policy_optimizer.step()
      
      return policy_loss