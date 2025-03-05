import gym
from gym import spaces


import time
import numpy as np
from collections import deque
from numpy import linalg as LA
from numpy import pi, cos, sin, tan, sqrt
from math import atan2, asin

D2R = pi / 180

class Kinematics3d_Env(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(Kinematics3d_Env, self).__init__()
        self.reward                     = 0
        self.episode                    = 0
        self.success_epi                = 0
        self.noise                      = 0.001
        self.dist_max                   = 5
        self.control_gain               = 400
        self.obs_stack                  = 1
        
        '''
        Missile : N, E, D, u, v, w, roll, pitch, yaw
        Target  : N, E, D, u, v, w, roll, pitch, yaw
        '''
        self.state_dim                  = 9
        self.obs_dim                    = 10
        self.act_dim                    = 2
        self.grav                       = 9.806
        
        self.step_hz                    = 100
        self.dt                         = 1/(self.step_hz*np.ones(1))
        
        # Gym Setting
        self.observation_space = spaces.Box(low=-pi, high = pi, shape=(self.obs_stack, self.obs_dim), dtype=np.float32)
        self.action_space = spaces.Box(low = -1, high = 1, shape=(self.act_dim,), dtype=np.float32)

    def reset(self, options: dict = {}):    
        self.reward                     = 0
        self.count                      = 0
        self.state_save                 = []
        self.action_save                = []
        self.los_save                   = []
        self.obs_queue                  = deque()
        self.min_dist                   = np.inf
        
        self.M_states                   = np.zeros(self.state_dim)
        self.T_states                   = np.zeros(self.state_dim)
        self.next_M_states              = np.zeros(self.state_dim)
        self.next_T_states              = np.zeros(self.state_dim)

        self.M_states[0]                = 0
        self.M_states[1]                = 0
        self.M_states[2]                = 0
        self.M_states[3]                = 900
        self.M_states[7] = options.get('M_pitch', 35) * D2R
        self.M_states[8] = options.get('M_yaw', 45) * D2R
        
        self.T_states[0]                = 5e3
        self.T_states[1]                = 5e3
        self.T_states[2]                = -5e3
        self.T_states[3]                = 700
        if options.get('T_pitch') is None or options.get('T_yaw') is None:
            if np.random.rand() < (441 / (441 + 256)):  # head-on
                self.T_states[7] = -70 + np.random.randint(21) * 3
                self.T_states[8] = -165 + np.random.randint(21) * 3
            else:  # chase
                self.T_states[7] = 10 + np.random.randint(16) * 4
                self.T_states[8] = 15 + np.random.randint(16) * 4
        else:
            self.T_states[7] = options.get('T_pitch')
            self.T_states[8] = options.get('T_yaw')
        self.T_states[7:9] = self.T_states[7:9] * D2R

        self.tau = options.get('tau', np.random.randint(1, 4) * 0.1)
        self.ap_M = 0
        self.aq_M = 0
        self.ar_M = 0

        Pos_M                           = self.M_states[:3]
        Pos_T                           = self.T_states[:3]
        self.Initial_Range              = LA.norm(Pos_M - Pos_T)

        obs = self.get_obs(self.M_states, self.T_states, self.Initial_Range)
        if self.obs_stack == 1:
            return obs, {'Mstates': self.M_states, 'Tstates': self.T_states}
        else:
            for _ in range(self.obs_stack):
                self.obs_queue.append(obs)
            return self.obs_queue, {'Mstates': self.M_states, 'Tstates': self.T_states}

    def get_diff(self, action, Mstates, Tstates):
        
        pos_M = Mstates[:3]; vel_M = Mstates[3:6]; eul_M = Mstates[6:]
        x_M = pos_M[0]; y_M = pos_M[1]; z_M = pos_M[2]
        u_M = vel_M[0]; v_M = vel_M[1]; w_M = vel_M[2]; V_tot_M = LA.norm(vel_M)
        phi_M = eul_M[0]; theta_M = eul_M[1]; psi_M = eul_M[2]

        pos_T = Tstates[:3]; vel_T = Tstates[3:6]; eul_T = Tstates[6:]
        x_T = pos_T[0]; y_T = pos_T[1]; z_T = pos_T[2]
        u_T = vel_T[0]; v_T = vel_T[1]; w_T = vel_T[2]; V_tot_T = LA.norm(vel_T)
        phi_T = eul_T[0]; theta_T = eul_T[1]; psi_T = eul_T[2]
        
        phi_M = self.ang_check(phi_M)
        theta_M = self.ang_check(theta_M)
        psi_M = self.ang_check(psi_M)
        
        phi_T = self.ang_check(phi_T)
        theta_T = self.ang_check(theta_T)
        psi_T = self.ang_check(psi_T)
        
        x_M_dot = u_M*cos(theta_M)*cos(psi_M) + v_M*(-cos(phi_M)*sin(psi_M)+sin(phi_M)*sin(theta_M)*cos(psi_M)) + w_M*(sin(phi_M)*sin(psi_M)+cos(phi_M)*sin(theta_M)*cos(psi_M))
        y_M_dot = u_M*cos(theta_M)*sin(psi_M) + v_M*(cos(phi_M)*cos(psi_M)+sin(phi_M)*sin(theta_M)*sin(psi_M)) + w_M*(-sin(phi_M)*cos(psi_M)+cos(phi_M)*sin(theta_M)*sin(psi_M))
        z_M_dot = -u_M*sin(theta_M) + v_M*sin(phi_M)*cos(theta_M) + w_M*cos(phi_M)*cos(theta_M)

        x_T_dot = u_T*cos(theta_T)*cos(psi_T) + v_T*(-cos(phi_T)*sin(psi_T)+sin(phi_T)*sin(theta_T)*cos(psi_T)) + w_T*(sin(phi_T)*sin(psi_T)+cos(phi_T)*sin(theta_T)*cos(psi_T))
        y_T_dot = u_T*cos(theta_T)*sin(psi_T) + v_T*(cos(phi_T)*cos(psi_T)+sin(phi_T)*sin(theta_T)*sin(psi_T)) + w_T*(-sin(phi_T)*cos(psi_T)+cos(phi_T)*sin(theta_T)*sin(psi_T))
        z_T_dot = -u_T*sin(theta_T) + v_T*sin(phi_T)*cos(theta_T) + w_T*cos(phi_T)*cos(theta_T)
        
        ax_M = 0.0
        if (self.tau is None) or (abs(self.tau) < 1e-6):
            ap_M = V_tot_M * (- 50 * (phi_M - 0))
            aq_M = action[0]
            ar_M = action[1]
        else:  # account for time delay
            ap_M_dot = - 1 / self.tau * (self.ap_M - V_tot_M * (- 50 * (phi_M - 0)))
            aq_M_dot = - 1 / self.tau * (self.aq_M - action[0])
            ar_M_dot = - 1 / self.tau * (self.ar_M - action[1])
            self.ap_M = self.ap_M + ap_M_dot * self.dt
            self.aq_M = self.aq_M + aq_M_dot * self.dt
            self.ar_M = self.ar_M + ar_M_dot * self.dt
            ap_M = self.ap_M
            aq_M = self.aq_M
            ar_M = self.ar_M
        p_M = ap_M/V_tot_M; q_M = aq_M/V_tot_M; r_M = ar_M/V_tot_M
        
        ax_T= 0.0; ap_T = 0.0; aq_T = 5.0*self.grav; ar_T = 5.0*self.grav
        p_T = ap_T/V_tot_T; q_T = aq_T/V_tot_T; r_T = ar_T/V_tot_T

        u_M_dot = r_M * v_M - q_M * w_M + ax_M
        v_M_dot = 0
        w_M_dot = 0

        phi_M_dot = p_M + tan(theta_M) * (q_M * sin(phi_M) + r_M * cos(phi_M))
        theta_M_dot = q_M * cos(phi_M) - r_M * sin(phi_M)
        psi_M_dot = (q_M * sin(phi_M) + r_M * cos(phi_M)) / cos(theta_M)

        u_T_dot = 0; v_T_dot = 0; w_T_dot = 0
        phi_T_dot = p_T + tan(theta_T)*(q_T*sin(phi_T)+r_T*cos(phi_T))
        theta_T_dot = q_T*cos(phi_T) - r_T*sin(phi_T)
        psi_T_dot = (q_T*sin(phi_T) + r_T*cos(phi_T))/cos(theta_T)

        _Mstates       = np.zeros(self.state_dim)
        _Tstates       = np.zeros(self.state_dim)
       
        _Mstates[0]    = x_M_dot 
        _Mstates[1]    = y_M_dot 
        _Mstates[2]    = z_M_dot 
        _Mstates[3]    = u_M_dot 
        _Mstates[4]    = v_M_dot 
        _Mstates[5]    = w_M_dot 
        _Mstates[6]    = phi_M_dot 
        _Mstates[7]    = theta_M_dot 
        _Mstates[8]    = psi_M_dot 

        _Tstates[0]    = x_T_dot 
        _Tstates[1]    = y_T_dot 
        _Tstates[2]    = z_T_dot 
        _Tstates[3]    = u_T_dot 
        _Tstates[4]    = v_T_dot 
        _Tstates[5]    = w_T_dot 
        _Tstates[6]    = phi_T_dot 
        _Tstates[7]    = theta_T_dot 
        _Tstates[8]    = psi_T_dot 
        
        return _Mstates, _Tstates
    
    def get_reward(self ,Mstates, Tstates, prev_Mstates, prev_Tstates):
        
        pos_M = Mstates[:3]; vel_M = Mstates[3:6]; eul_M = Mstates[6:]
        x_M = pos_M[0]; y_M = pos_M[1]; z_M = pos_M[2]; h_M = -z_M
        u_M = vel_M[0]; v_M = vel_M[1]; w_M = vel_M[2]; V_tot_M = LA.norm(vel_M)
        phi_M = eul_M[0]; theta_M = eul_M[1]; psi_M = eul_M[2]

        pos_T = Tstates[:3]; vel_T = Tstates[3:6]; eul_T = Tstates[6:]
        x_T = pos_T[0]; y_T = pos_T[1]; z_T = pos_T[2]; h_T = -z_T
        u_T = vel_T[0]; v_T = vel_T[1]; w_T = vel_T[2]; V_tot_T = LA.norm(vel_T)
        phi_T = eul_T[0]; theta_T = eul_T[1]; psi_T = eul_T[2]

        prev_Pos_M = prev_Mstates[:3]
        prev_Pos_T = prev_Tstates[:3]

        prev_Range  = LA.norm(prev_Pos_T-prev_Pos_M) 
        Range       = LA.norm(pos_T-pos_M)
        LOS_t       = atan2(y_T-y_M,x_T-x_M)
        LOS_g       = asin((h_T-h_M)/Range)

        rotm_B2I = self.eul2rotm(psi_M, theta_M, phi_M)
        rotm_L2I = self.eul2rotm(LOS_t, LOS_g, 0)
        rotm_L2B = np.dot(np.transpose(rotm_B2I),rotm_L2I)
        eul_look = self.rotm2eul(np.transpose(rotm_L2B))
        phi_look = eul_look[2]; theta_look = eul_look[1]; psi_look = eul_look[0]          # 표적에 대한 탐색기 시야각

        rotm_T2I = self.eul2rotm(psi_T, theta_T, phi_T)
        rotm_L2I = self.eul2rotm(LOS_t, LOS_g, 0)
        rotm_L2T = np.dot(np.transpose(rotm_T2I),rotm_L2I)
        eul_look_T = self.rotm2eul(np.transpose(rotm_L2T))
        phi_look_T = eul_look_T[2]; theta_look_T = eul_look_T[1]; psi_look_T = eul_look_T[0]

        Range_dot = ( V_tot_T*cos(theta_look_T)*cos(psi_look_T) - V_tot_M*cos(theta_look)*cos(psi_look) ) 
        LOS_t_dot = ( V_tot_T*cos(theta_look_T)*sin(psi_look_T) - V_tot_M*cos(theta_look)*sin(psi_look) )/Range 
        LOS_g_dot = ( V_tot_T*sin(theta_look_T) - V_tot_M*sin(theta_look) )/Range 

        self.los_save.append([LOS_t, LOS_g, LOS_t_dot, LOS_g_dot])

        Vr        = V_tot_T * cos(psi_T-LOS_t) - V_tot_M * cos(psi_M-LOS_t)
        Vl        = V_tot_T * sin(psi_T-LOS_t) - V_tot_M * sin(psi_M-LOS_t)
        sqrt_     = sqrt(Vr**2+Vl**2) 
        zem1      = (Range*Vl)/sqrt_ 

        Vr        = V_tot_T * cos(theta_T-LOS_g) - V_tot_M * cos(theta_M-LOS_g) 
        Vl        = V_tot_T * sin(theta_T-LOS_g) - V_tot_M * sin(theta_M-LOS_g)
        sqrt_     = sqrt(Vr**2+Vl**2) 
        zem2      = (Range*Vl)/sqrt_ 

        zem       = sqrt(zem1**2+zem2**2)

        aa        = 100000.0
        
        if Range <= self.dist_max:
            reward = 100
        elif Range > prev_Range:
            reward = -100
        else:
            reward    = - (zem/aa)**2
        
        return reward, (Range_dot, LOS_t, LOS_g, LOS_t_dot, LOS_g_dot)
    
    def get_obs(self, Mstates, Tstates, initial_Range):

        pos_M = Mstates[:3]; vel_M = Mstates[3:6]; eul_M = Mstates[6:]
        x_M = pos_M[0]; y_M = pos_M[1]; z_M = pos_M[2]; h_M = -z_M
        u_M = vel_M[0]; v_M = vel_M[1]; w_M = vel_M[2]; V_tot_M = LA.norm(vel_M)
        phi_M = eul_M[0]; theta_M = eul_M[1]; psi_M = eul_M[2]

        pos_T = Tstates[:3]; vel_T = Tstates[3:6]; eul_T = Tstates[6:]
        x_T = pos_T[0]; y_T = pos_T[1]; z_T = pos_T[2]; h_T = -z_T
        u_T = vel_T[0]; v_T = vel_T[1]; w_T = vel_T[2]; V_tot_T = LA.norm(vel_T)
        phi_T = eul_T[0]; theta_T = eul_T[1]; psi_T = eul_T[2]
        
        Range       = LA.norm(pos_T-pos_M)
        LOS_t       = atan2(y_T-y_M,x_T-x_M)
        LOS_g       = asin((h_T-h_M)/Range)
        
        rotm_B2I = self.eul2rotm(psi_M, theta_M, phi_M)
        rotm_L2I = self.eul2rotm(LOS_t, LOS_g, 0)
        rotm_L2B = np.dot(np.transpose(rotm_B2I),rotm_L2I)
        eul_look = self.rotm2eul(np.transpose(rotm_L2B))
        phi_look = eul_look[2]; theta_look = eul_look[1]; psi_look = eul_look[0]
        
        rotm_T2I = self.eul2rotm(psi_T, theta_T, phi_T)
        rotm_L2I = self.eul2rotm(LOS_t, LOS_g, 0)
        rotm_L2T = np.dot(np.transpose(rotm_T2I),rotm_L2I)
        eul_look_T = self.rotm2eul(np.transpose(rotm_L2T))
        phi_look_T = eul_look_T[2]; theta_look_T = eul_look_T[1]; psi_look_T = eul_look_T[0]

        Range_dot = ( V_tot_T*cos(theta_look_T)*cos(psi_look_T) - V_tot_M*cos(theta_look)*cos(psi_look) ) 
        LOS_t_dot = ( V_tot_T*cos(theta_look_T)*sin(psi_look_T) - V_tot_M*cos(theta_look)*sin(psi_look) )/Range 
        LOS_g_dot = ( V_tot_T*sin(theta_look_T) - V_tot_M*sin(theta_look) )/Range 

        # Normalization -pi ~ pi
        NN_input    = np.zeros(self.obs_dim)
        
        NN_input[0] = phi_look
        NN_input[1] = theta_look
        NN_input[2] = psi_look
        NN_input[3] = LOS_t
        NN_input[4] = LOS_g
        NN_input[5] = np.clip(LOS_t_dot, -np.pi/2, np.pi/2)
        NN_input[6] = np.clip(LOS_g_dot, -np.pi/2, np.pi/2)
        NN_input[7] = Range / initial_Range
        NN_input[8] = Range_dot / 1000
        NN_input[9] = 0 if self.tau is None else self.tau
        
        # NN_input = np.random.normal(loc = 1.0, scale = self.noise, size = self.obs_dim) * NN_input
        return NN_input.astype(np.float32)
    
    def get_rk4(self, Mstates, Tstates ,action):
        action[0] = max(min(action[0], 1),-1) # saturation
        action[1] = max(min(action[1], 1),-1) # saturation
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
        
        prev_Pos_M = self.M_states[:3]
        prev_Pos_T = self.T_states[:3]
        prev_Range = LA.norm(prev_Pos_T - prev_Pos_M)
        
        next_Mstates, next_Tstates = self.get_rk4(prev_Mstates, prev_Tstates, action)

        # State Save
        state = np.concatenate((next_Mstates,next_Tstates), axis=None)
        self.state_save.append(state)
        self.action_save.append(action)
        
        self.M_states = next_Mstates
        self.T_states = next_Tstates
        
        # Observation
        now_obs = self.get_obs(self.M_states, self.T_states, self.Initial_Range)
        self.obs_queue.append(now_obs)
        self.obs_queue.popleft()
        obs = self.obs_queue
        
        self.reward, labels = self.get_reward(self.M_states, self.T_states, prev_Mstates, prev_Tstates)
        
        Pos_M = self.M_states[:3]
        Pos_T = self.T_states[:3]
        Range = LA.norm(Pos_T - Pos_M)
        
        if Range < self.min_dist:
            self.min_dist = Range
        
        # if self.count%(self.step_hz) == 0.0:
        #     print("Epi: {} | Step: {:.3f} | Range: {:.3f} | Reward: {:.3f}".format(self.episode, self.count, Range, self.reward))

        if Range <= self.dist_max:
            success = True
            self.success_epi += 1
            # print("==========================================")
            # print("            Intercept Success             ")
            # print("==========================================")
            
        if success or (self.count > 5000) or (Range > prev_Range):
            # print("Epi: {} | Step: {:.3f} | Range: {:.3f} | Reward: {:.3f}".format(self.episode, self.count, Range, self.reward))
            done = True
            self.episode += 1
            # print("==========================================")
            # print("    Episode Done | success/episode :"+str(self.success_epi)+"/"+str(self.episode))
            # print("==========================================")
        
        info = {
            'success' : success,
            'Mstates' : self.M_states,
            'Tstates' : self.T_states,
            'min_dist' : self.min_dist,
            'r_dot': labels[0],
            'LOS_t': labels[1],
            'LOS_g': labels[2],
            'LOS_t_dot': labels[3],
            'LOS_g_dot': labels[4],
        }

        self.count += 1
        if self.obs_stack == 1:
            return now_obs, self.reward, done, info
        else:
            return obs, self.reward, done, info
        
    # ==================== Matlab Functions ====================

    def eul2rotm(self, psi_M, theta_M, phi_M):
        
        R_x = np.array([[1,         0,               0           ],
                        [0,         cos(phi_M),      -sin(phi_M) ],
                        [0,         sin(phi_M),      cos(phi_M)  ]
                        ])

        R_y = np.array([[cos(theta_M),    0,      sin(theta_M)  ],
                        [0,               1,      0             ],
                        [-sin(theta_M),   0,      cos(theta_M)  ]
                        ])

        R_z = np.array([[cos(psi_M),    -sin(psi_M),    0],
                        [sin(psi_M),    cos(psi_M),     0],
                        [0,             0,              1]
                        ])

        R = np.dot(R_z, np.dot( R_y, R_x ))
        return R        

    # Checks if a matrix is a valid rotation matrix.
    def isRotationMatrix(R) :
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    def rotm2eul(self, R) :

        # assert(isRotationMatrix(R))

        sy = sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

        singular = sy < 1e-6

        if  not singular :
            x = atan2(R[2,1] , R[2,2])
            y = atan2(-R[2,0], sy)
            z = atan2(R[1,0], R[0,0])
        else :
            x = atan2(-R[1,2], R[1,1])
            y = atan2(-R[2,0], sy)
            z = 0

        return np.array([z, y, x])        
    
    def ang_check(self, angle):
        if angle > np.pi:
            angle = angle - 2 * np.pi
        elif angle < - np.pi:
            angle = angle + 2 * np.pi
        else:
            angle = angle
            
        return angle