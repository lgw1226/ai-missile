import gym
from gym import spaces

import numpy as np
from collections import deque
from numpy import linalg as LA
from numpy import pi, cos, sin, sqrt
from math import atan2

D2R = pi / 180

class Kinematics2d_Env(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self): 
        super(Kinematics2d_Env, self).__init__()
        self.reward                     = 0
        self.episode                    = 0
        self.success_epi                = 0
        self.noise                      = 0.001
        self.control_gain               = 200

        '''
        Head-on scenario
        Missile : x, z, theta
        Target  : x, z, theta
        Obs     : r_dot, los, los_dot
        '''
        self.state_dim                  = 3
        self.obs_dim                    = 3
        self.act_dim                    = 1
        self.M_V                        = 500
        self.T_V                        = 200
        self.grav                       = 9.806

        self.step_hz                    = 100
        self.dt                         = 1/(self.step_hz*np.ones(1))
        
        # Gym setting
        self.observation_space = spaces.Box(low=-pi, high = pi, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low = -1, high = 1, shape=(self.act_dim,), dtype=np.float32)
        
    def reset(self):
        self.reward                     = 0
        self.count                      = 0
        self.los_save                   = []
        self.state_save                 = []
        self.action_save                = []
        self.min_dist                   = np.inf

        self.M_states                   = np.zeros(self.state_dim)
        self.T_states                   = np.zeros(self.state_dim)
        self.next_M_states              = np.zeros(self.state_dim)
        self.next_T_states              = np.zeros(self.state_dim)
        
        # Learning ====================================
        self.M_states[0]                = 0.0
        self.M_states[1]                = 0.0
        self.M_states[2]                = np.random.randint(0,5,1) * 10 * D2R
        
        self.T_states[0]                = 2e3
        self.T_states[1]                = 0.0
        self.T_states[2]                = np.random.randint(9,19,1) * 10 * D2R

        # Scenario ====================================
        # self.M_states[0]                = 0
        # self.M_states[1]                = 0
        # self.M_states[2]                = np.random.randint(0,41,1) * D2R

        # self.T_states[0]                = 2e3
        # self.T_states[1]                = 0
        # self.T_states[2]                = np.random.randint(90,181,1) * D2R

        Pos_M                           = self.M_states[:2]
        Pos_T                           = self.T_states[:2]
        self.Initial_Range              = LA.norm(Pos_M - Pos_T)
            
        print("=======================================================================")
        print("gamma_M : {:.1f} deg | gamma_T : {:.1f} deg".format(self.M_states[2]/D2R, self.T_states[2]/D2R))
        print("=======================================================================")
        
        obs = self.get_obs(self.M_states, self.T_states, self.Initial_Range)

        return obs

    def get_diff(self, action, Mstates, Tstates):

        M_V     = self.M_V
        M_Pos   = Mstates[:2]
        M_theta = Mstates[2]
        
        T_V   = self.T_V
        T_Pos = Tstates[:2]
        T_theta = Tstates[2]

        M_x_dot           = M_V*cos(M_theta)
        M_z_dot           = M_V*sin(M_theta)
        M_theta_dot       = action/M_V
        # M_theta_dot       = (action - self.grav*cos(M_theta))/V_M
        
        T_x_dot           = T_V*cos(T_theta)
        T_z_dot           = T_V*sin(T_theta)
        T_theta_dot       = 0

        _Mstates       = np.zeros(3)
        _Tstates       = np.zeros(3)
        
        _Mstates[0]    = M_x_dot 
        _Mstates[1]    = M_z_dot 
        _Mstates[2]    = M_theta_dot 
        
        _Tstates[0]    = T_x_dot 
        _Tstates[1]    = T_z_dot 
        _Tstates[2]    = T_theta_dot 
        
        return _Mstates, _Tstates
    
    def get_reward(self, Mstates, Tstates, prev_Mstates, prev_Tstates):
        
        M_V   = self.M_V
        M_Pos = Mstates[:2]
        M_theta = Mstates[2]
        
        T_V   = self.T_V
        T_Pos = Tstates[:2]
        T_theta = Tstates[2]
        
        prev_Pos_M = prev_Mstates[:2]
        prev_Pos_T = prev_Tstates[:2]

        prev_Range  = LA.norm(prev_Pos_T-prev_Pos_M) 
        Range       = LA.norm(T_Pos-M_Pos)
        LOS         = atan2(T_Pos[1]-M_Pos[1], T_Pos[0]-M_Pos[0])
        Range_dot   = T_V*cos(T_theta-LOS) - M_V*cos(M_theta-LOS)
        LOS_dot     = (T_V*sin(T_theta-LOS) - M_V*sin(M_theta-LOS))/Range

        self.los_save.append([LOS, LOS_dot])
        Vr          = T_V * cos(T_theta-LOS) - M_V * cos(M_theta-LOS)
        Vl          = Range*LOS_dot
        zem         = Range * Vl / sqrt(pow(Vr,2) + pow(Vl,2))
        
        aa          = 10000.0
        
        if Range > prev_Range:
            reward = -100
        elif Range < 3:
            reward = 100
        else:
            reward    = - (zem/aa)**2

        return reward

    def get_obs(self, Mstates, Tstates, initial_range):

        M_V   = self.M_V
        M_Pos = Mstates[:2]
        M_theta = Mstates[2]
        
        T_V   = self.T_V
        T_Pos = Tstates[:2]
        T_theta = Tstates[2]

        Range       = LA.norm(T_Pos-M_Pos)
        LOS         = atan2(T_Pos[1]-M_Pos[1], T_Pos[0]-M_Pos[0])
        Range_dot   = T_V*cos(T_theta-LOS) - M_V*cos(M_theta-LOS)
        LOS_dot     = (T_V*sin(T_theta-LOS) - M_V*sin(M_theta-LOS))/Range
        
        NN_input    = np.zeros(self.obs_dim)

        # NN_input[0] = Range / initial_range
        # NN_input[1] = Range_dot
        # NN_input[2] = LOS
        # NN_input[3] = LOS_dot

        NN_input[0] = Range_dot
        NN_input[1] = LOS
        NN_input[2] = LOS_dot

        # Add noise
        # NN_input = np.random.normal(loc = 1.0, scale = self.noise, size = self.obs_dim) * NN_input

        return NN_input.astype(np.float32)

    def get_rk4(self, Mstates, Tstates ,action):
        action = action * self.control_gain 
        
        prev_Mstates = Mstates
        prev_Tstates = Tstates
        
        Mstates_dot = np.zeros((self.state_dim,4))
        Tstates_dot = np.zeros((self.state_dim,4))
        
        for j in range(4):
            Mstates_dot[:,j], Tstates_dot[:,j] = self.get_diff(action, prev_Mstates, prev_Tstates)

            if j < 2:
               prev_Mstates = self.M_states + self.dt * Mstates_dot[:,j] / 2 
               prev_Tstates = self.T_states + self.dt * Tstates_dot[:,j] / 2 
            else:
               prev_Mstates = self.M_states + self.dt * Mstates_dot[:,j] 
               prev_Tstates = self.T_states + self.dt * Tstates_dot[:,j] 

        next_M_states = (self.M_states + self.dt/6*(Mstates_dot[:,0]+2*Mstates_dot[:,1]+2*Mstates_dot[:,2]+Mstates_dot[:,3]))
        next_T_states = (self.T_states + self.dt/6*(Tstates_dot[:,0]+2*Tstates_dot[:,1]+2*Tstates_dot[:,2]+Tstates_dot[:,3]))
    
        return next_M_states, next_T_states
    
    def step(self, action):
        success = False
        done = False
        
        prev_Mstates = self.M_states
        prev_Tstates = self.T_states
        
        prev_Pos_M = self.M_states[:2]
        prev_Pos_T = self.T_states[:2]
        prev_Range = LA.norm(prev_Pos_T - prev_Pos_M)
        
        next_Mstates, next_Tstates = self.get_rk4(prev_Mstates, prev_Tstates, action)
        
        # State Save
        state = np.concatenate((next_Mstates,next_Tstates), axis=None)
        self.state_save.append(state)
        self.action_save.append(action*self.control_gain)
            
        self.M_states = next_Mstates
        self.T_states = next_Tstates
        
        # Observation
        obs = self.get_obs(self.M_states, self.T_states, self.Initial_Range)
        
        self.reward = self.get_reward(self.M_states, self.T_states, prev_Mstates, prev_Tstates)
        
        Pos_M = self.M_states[:2]
        Pos_T = self.T_states[:2]
        Range = LA.norm(Pos_T - Pos_M)
        
    
        if Range < self.min_dist:
            self.min_dist = Range
        
        if self.count%(self.step_hz) == 0.0:
            print("Epi: {} | Step: {:.3f} | Range: {:.3f} | Reward: {:.3f}".format(self.episode, self.count, Range, self.reward))
            # print("OBS: {}".format(obs))
            
        if Range <= 3:
            success = True
            self.success_epi += 1
            print("==========================================")
            print("            Intercept Success             ")
            print("==========================================")
            
        if success or (self.count > 5000) or (Range > prev_Range):
            print("Epi: {} | Step: {:.3f} | Range: {:.3f} | Reward: {:.3f}".format(self.episode, self.count, Range, self.reward))
            done = True
            self.episode += 1
            print("==========================================")
            print("    Episode Done | success/episode :"+str(self.success_epi)+"/"+str(self.episode))
            print("==========================================")
        
        info = {
            'success' : success,
            'Mstates' : self.M_states,
            'Tstates' : self.T_states,
            'min_dist' : self.min_dist
        }

        self.count += 1

        return obs, self.reward, done, info

        