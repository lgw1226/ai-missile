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

 

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

class ConvPolicy(nn.Module):

    def __init__(self, h, w, act_dim, act_limit, device):
        super(ConvPolicy, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2) #sequential3
        #self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        #self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        #self.bn3 = nn.BatchNorm2d(32)
        self.device = device
        self.act_dim = act_dim
        self.act_limit = act_limit

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.fc = nn.Sequential(
           nn.Linear(linear_input_size, 256), nn.ReLU(),
           nn.Linear(256, 256), nn.ReLU(),
           nn.Linear(256, act_dim)
        )
        self.apply(weight_init)


    def forward(self, obs, compute_pi=True, compute_log_pi=True):
        #x = x.to(device)
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)


        return torch.tanh(x) * self.act_limit



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
      self.obs_dim = env.obs_dim
      self.act_dim = env.act_dim
      self.act_limit = 400#env.act_limit #jay_check
      self.act_noise = act_noise
      self.hidden_sizes = hidden_sizes
      self.policy_lr = policy_lr
      self.qf_lr = qf_lr
      self.gradient_clip_policy = gradient_clip_policy
      self.gradient_clip_qf = gradient_clip_qf

      # Main network
      if self.args.render:
         self.policy = ConvPolicy(self.args.image_obs_dim[1], self.args.image_obs_dim[2], self.act_dim, self.act_limit, device).to(self.device)
      else:
         self.policy = MLP(self.obs_dim, self.act_dim, self.act_limit, 
                                       hidden_sizes=self.hidden_sizes, 
                                       output_activation=torch.tanh,
                                       use_actor=True).to(self.device)
      # Create optimizers
      self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.policy_lr)
      self.loss = nn.MSELoss()

   def select_action(self, state, evaluate=False):
      if not torch.is_tensor(state):
         state = torch.from_numpy(state)
      NN_input = torch.unsqueeze(state.type(torch.FloatTensor), dim=0)
      #normalize image
      NN_input = NN_input/255.
      action = self.policy(NN_input.to(self.device))[0].detach().cpu().numpy()
      if(action.ndim == 2):
         action = np.squeeze(action, axis=0)
      #if not evaluate:
      #   action += self.act_noise * np.random.randn(self.act_dim)
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
         self.policy.load_state_dict(checkpoint['policy_state_dict'])
         self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])

         if evaluate:
            self.policy.eval()
         else:
            self.policy.train()