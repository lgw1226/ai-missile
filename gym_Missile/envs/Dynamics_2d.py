import gym
from gym import spaces

import time
import numpy as np
from collections import deque
from math import atan2, isnan
from numpy import linalg as LA
from numpy import pi, cos, sin, sqrt

D2R = pi / 180

class Dynamics2d_Env(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(Dynamics2d_Env, self).__init__()
        # Simulation Setting
        self.reward                     = 0
        self.episode                    = 0
        self.success_epi                = 0
        self.noise                      = 0.001
        self.control_gain               = 30 * D2R
        self.target_gain                = -5 # default : -5
        self.obs_stack                  = 1
        

        '''
        Missile : X, Z, alpha, theta, q, delta

        x: x position
        z: z position
        alpha: angle of attack
        theta: pitch angle
        q: pitch rate
        delta: control surface deflection

        Target  : X, Z, gamma
        '''



        self.state_dim                  = 6
        self.T_state_dim                = 3
        self.obs_dim                    = 5
        self.act_dim                    = 1
        
        # Parameters
        self.L_alpha_B                  = 1190
        self.L_delta                    = 80
        self.M_alpha_B                  = -234
        self.M_delta                    = 160
        self.M_q                        = -5
        self.tau_q                      = 0.02
        self.sat_angle                  = 30 * D2R
        self.M_V                        = 500
        self.T_V                        = 200
        self.grav                       = 9.806

        self.step_hz                    = 100
        self.dt                         = 1/(self.step_hz*np.ones(1))
                
        # Gym setting
        self.observation_space = spaces.Box(low=-pi, high = pi, shape=(self.obs_stack, self.obs_dim), dtype=np.float32)
        self.action_space = spaces.Box(low = -1, high = 1, shape=(self.act_dim,), dtype=np.float32)
        
    def reset(self, options={}):
        self.reward                     = 0
        self.count                      = 0
        self.state_save                 = []
        self.action_save                = []
        self.los_save                   = []
        self.los_dot_save               = []
        self.am_save                    = []
        self.obs_queue                  = deque()
        self.min_dist                   = np.inf
        
        self.M_states                   = np.zeros(self.state_dim)
        self.T_states                   = np.zeros(self.T_state_dim)
        self.next_M_states              = np.zeros(self.state_dim)
        self.next_T_states              = np.zeros(self.T_state_dim)
        
        self.M_states[0]                = 0
        self.M_states[1]                = 0
        self.M_states[2]                = 0
        # self.M_states[2]                = np.random.randint(0, 6, 1) * D2R 
        # self.M_states[3]                = np.random.randint(1,8,1) * 10 * D2R 
        self.M_states[3]                = np.random.randint(0, 41, 1) * D2R
        # self.M_states[2]                = 0 
        # self.M_states[3]                = 50 * D2R
        self.M_states[4]                = 0
        self.M_states[5]                = 0
        
        self.T_states[0]                = 5000
        self.T_states[1]                = 0
        # self.T_states[2]                = np.random.randint(11,18,1) * 10 * D2R 
        self.T_states[2]                = np.random.randint(90, 181, 1) * D2R 
        # self.T_states[2]                = 150 * D2R

        # if option is given, set initial conditions as following
        self.M_states[3] = options.get('theta_m', self.M_states[2] / D2R) * D2R
        self.T_states[2] = options.get('theta_t', self.T_states[2] / D2R) * D2R
        self.success_dist = options.get('success_dist', 3)

        # print("=======================================================================")
        # print("Alpha_M : {:.1f} deg | Theta_M : {:.1f} deg | Gamma_T : {:.1f} deg".format(self.M_states[2]/D2R, self.M_states[3]/D2R, self.T_states[2]/D2R))
        # print("=======================================================================")
        
        obs = self.get_obs(self.M_states, self.T_states)
        info = {'Mstates' : self.M_states, 'Tstates' : self.T_states}
        if self.obs_stack == 1:
            return obs, info
        else:
            for _ in range(self.obs_stack):
                self.obs_queue.append(obs)
            array = np.array(self.obs_queue)
            return array, info

    def get_diff(self, action, Mstates, Tstates):
        M_V             = self.M_V
        M_Pos           = Mstates[:2]
        M_alpha         = Mstates[2]
        M_theta         = Mstates[3]
        M_q             = Mstates[4]
        M_delta_q       = Mstates[5]
        M_gamma         = M_theta - M_alpha 
        
        T_V             = self.T_V
        T_Pos           = Tstates[:2]
        T_gamma         = Tstates[2]    
        
        M_alpha = self.ang_check(M_alpha)
        M_gamma = self.ang_check(M_gamma)
        T_gamma = self.ang_check(T_gamma)

        Lift_mass = self.L_alpha_B*max(min(M_alpha,self.sat_angle),-self.sat_angle) \
                    + self.L_delta*max(min(M_alpha+M_delta_q,self.sat_angle),-self.sat_angle)
        M_I       = self.M_alpha_B*max(min(M_alpha,self.sat_angle),-self.sat_angle) \
                    + self.M_q*M_q + self.M_delta*max(min(M_alpha+M_delta_q,self.sat_angle),-self.sat_angle)

        # state update
        M_x_dot         = self.M_V*cos(M_gamma)
        M_z_dot         = -self.M_V*sin(M_gamma)
        alpha_M_dot     = M_q - Lift_mass/M_V
        M_q_dot         = M_I
        delta_q_dot     = (action - M_delta_q)/self.tau_q

        a_T             = self.target_gain * self.grav
        T_x_dot         = T_V*cos(T_gamma)
        T_z_dot         = -T_V*sin(T_gamma)
        T_gamma_dot     = a_T/T_V
    
        _Mstates       = np.zeros(self.state_dim)
        _Tstates       = np.zeros(self.T_state_dim)
        
        _Mstates[0]    = M_x_dot 
        _Mstates[1]    = M_z_dot 
        _Mstates[2]    = alpha_M_dot
        _Mstates[3]    = M_q           # M_theta_dot
        _Mstates[4]    = M_q_dot
        _Mstates[5]    = delta_q_dot
        
        _Tstates[0]    = T_x_dot 
        _Tstates[1]    = T_z_dot 
        _Tstates[2]    = T_gamma_dot 
        
        return _Mstates, _Tstates
    
    def get_reward(self, Mstates, Tstates, prev_Mstates, prev_Tstates):
        M_V             = self.M_V
        M_Pos           = Mstates[:2]
        M_alpha         = Mstates[2]
        M_theta         = Mstates[3]
        M_q             = Mstates[4]
        M_delta_q       = Mstates[5]
        M_gam           = M_theta - M_alpha 
        
        T_V             = self.T_V
        T_Pos           = Tstates[:2]
        T_gam           = Tstates[2]
        
        prev_Pos_M = prev_Mstates[:2]
        prev_Pos_T = prev_Tstates[:2]

        prev_Range  = LA.norm(prev_Pos_T-prev_Pos_M) 
        Range       = LA.norm(T_Pos-M_Pos)
        LOS         = atan2(-T_Pos[1]+M_Pos[1], T_Pos[0]-M_Pos[0])
        LOS_dot     = (T_V*sin(T_gam-LOS) - M_V*sin(M_gam-LOS))/Range

        Lift_mass = self.L_alpha_B*max(min(M_alpha,self.sat_angle),-self.sat_angle) \
                    + self.L_delta*max(min(M_alpha+M_delta_q,self.sat_angle),-self.sat_angle)
        self.am_save.append(Lift_mass)

        self.los_save.append(LOS)
        self.los_dot_save.append(LOS_dot)

        Vr          = T_V * cos(T_gam-LOS) - M_V * cos(M_gam-LOS)
        Vl          = Range*LOS_dot
        zem         = Range * Vl / sqrt(np.power(Vr,2) + np.power(Vl,2))
        
        aa          = 10000.0
        
        if Range > prev_Range:
            reward = -100
        else:
            zem_reward = (zem/aa)**2
            reward    = - zem_reward

        return reward

    def get_obs(self, Mstates, Tstates):
        M_V             = self.M_V
        M_Pos           = Mstates[:2]
        M_alpha         = Mstates[2]
        M_theta         = Mstates[3]
        M_q             = Mstates[4]
        M_delta_q       = Mstates[5]
        M_gam           = M_theta - M_alpha 
        
        T_V             = self.T_V
        T_Pos           = Tstates[:2]
        T_gam           = Tstates[2]  

        Range       = LA.norm(T_Pos-M_Pos)
        LOS         = atan2(-T_Pos[1]+M_Pos[1], T_Pos[0]-M_Pos[0])
        Range_dot   = T_V*cos(T_gam-LOS) - M_V*cos(M_gam-LOS)
        LOS_dot     = (T_V*sin(T_gam-LOS) - M_V*sin(M_gam-LOS))/Range
        
        NN_input    = np.zeros(self.obs_dim)

        NN_input[0] = sin(LOS - M_gam)
        NN_input[1] = cos(LOS - M_gam)
        NN_input[2] = LOS_dot
        NN_input[3] = M_q
        NN_input[4] = M_delta_q
        
        NN_input = np.random.normal(loc = 1.0, scale = self.noise, size = self.obs_dim) * NN_input

        return NN_input.astype(np.float32)
    
    
    def get_rk4(self, Mstates, Tstates ,action):
        action = max(min(action, 1),-1) # saturation
        action = action * self.control_gain
        
        prev_Mstates = Mstates
        prev_Tstates = Tstates
        
        Mstates_dot = np.zeros((self.state_dim,4))
        Tstates_dot = np.zeros((self.T_state_dim,4))
        
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
        self.action_save.append(action)
            
        self.M_states = next_Mstates
        self.T_states = next_Tstates

        # Observation
        now_obs = self.get_obs(self.M_states, self.T_states)
        self.obs_queue.append(now_obs)
        self.obs_queue.popleft()
        obs = self.obs_queue
        obs = np.array(obs)
        
        self.reward = self.get_reward(self.M_states, self.T_states, prev_Mstates, prev_Tstates)
        
        Pos_M = self.M_states[:2]
        Pos_T = self.T_states[:2]
        Range = LA.norm(Pos_T - Pos_M)
        
        
        if Range < self.min_dist:
            self.min_dist = Range
    
        # if self.count%(self.step_hz) == 0.0:
            # print("Epi: {} | Step: {:.3f} | Range: {:.3f} | Reward: {:.3f}".format(self.episode, self.count, Range, self.reward))
            # print("OBS: {}".format(obs))
            
        if Range <= self.success_dist:
            success = True
            self.success_epi += 1
            # print("==========================================")
            # print("            Intercept Success             ")
            # print("==========================================")
        
        max_episode_len = 2000
        if success or (self.count > max_episode_len) or (Range > prev_Range):
            # print("Epi: {} | Step: {:.3f} | Range: {:.3f} | Reward: {:.3f}".format(self.episode, self.count, Range, self.reward))
            done = True
            self.episode += 1
            # print("==========================================")
            # print("    Episode Done | success/episode :"+str(self.success_epi)+"/"+str(self.episode))
            # print("==========================================")
        if isnan(Range):
            # print("Epi: {} | Step: {:.3f} | Range: {:.3f} | Reward: {:.3f}".format(self.episode, self.count, Range, self.reward))
            done = True
            success = False
            self.episode += 1
            # print("==========================================")
            # print("    Episode Done | success/episode :"+str(self.success_epi)+"/"+str(self.episode))
            # print("==========================================")
        info = {
            'success' : success,
            'Mstates' : self.M_states,
            'Tstates' : self.T_states,
            'min_dist' : self.min_dist
        }
        
        self.count += 1
        if self.obs_stack == 1:
            return now_obs, self.reward, done, info
        else:
            return obs, self.reward, done, info
        
    def ang_check(self, angle):
        if angle > np.pi:
            angle = angle - 2 * np.pi
        elif angle < - np.pi:
            angle = angle + 2 * np.pi
        else:
            angle = angle
            
        return angle