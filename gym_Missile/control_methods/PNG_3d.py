import numpy as np
from math import pi, atan2, asin
from numpy import cos, sin
from numpy import linalg as LA

D2R = pi / 180

class PNG_3d:
    def __init__(self, env, gain: int = 5):
        super(PNG_3d, self).__init__()
        self.env                               = env
        self.grav                              = 9.806
        self.N_g                               = gain
        self.N_t                               = gain
        
    def control_input(self, Mstates, Tstates):
    
        pos_M = Mstates[:3]; vel_M = Mstates[3:6]; eul_M = Mstates[6:]
        x_M = pos_M[0]; y_M = pos_M[1]; z_M = pos_M[2]; h_M = -z_M
        u_M = vel_M[0]; v_M = vel_M[1]; w_M = vel_M[2]; V_tot_M = LA.norm(vel_M)
        phi_M = eul_M[0]; theta_M = eul_M[1]; psi_M = eul_M[2]

        pos_T = Tstates[:3]; vel_T = Tstates[3:6]; eul_T = Tstates[6:]
        x_T = pos_T[0]; y_T = pos_T[1]; z_T = pos_T[2]; h_T = -z_T
        u_T = vel_T[0]; v_T = vel_T[1]; w_T = vel_T[2]; V_tot_T = LA.norm(vel_T)
        phi_T = eul_T[0]; theta_T = eul_T[1]; psi_T = eul_T[2]

        Range                                  = LA.norm(pos_T-pos_M)
        LOS_t                                  = atan2(y_T-y_M,x_T-x_M)
        LOS_g                                  = asin((h_T-h_M)/Range)
        # LOS_t                                  = np.random.normal(loc = 1.0, scale = 0.1, size = 1) * LOS_t
        # LOS_g                                  = np.random.normal(loc = 1.0, scale = 0.1, size = 1) * LOS_g

        rotm_B2I                               = self.env.eul2rotm(psi_M, theta_M, phi_M)
        rotm_L2I                               = self.env.eul2rotm(LOS_t, LOS_g, 0)
        rotm_L2B                               = np.dot(np.transpose(rotm_B2I),rotm_L2I)
        eul_look                               = self.env.rotm2eul(np.transpose(rotm_L2B))
        phi_look                               = eul_look[2]
        theta_look                             = eul_look[1] 
        psi_look                               = eul_look[0] # 표적에 대한 탐색기 시야각

        rotm_T2I                               = self.env.eul2rotm(psi_T, theta_T, phi_T)
        rotm_L2I                               = self.env.eul2rotm(LOS_t, LOS_g, 0)
        rotm_L2T                               = np.dot(np.transpose(rotm_T2I),rotm_L2I)
        
        eul_look_T                             = self.env.rotm2eul(np.transpose(rotm_L2T))
        phi_look_T                             = eul_look_T[2]
        theta_look_T                           = eul_look_T[1]
        psi_look_T                             = eul_look_T[0]

        LOS_t_dot                              = ( V_tot_T*cos(theta_look_T)*sin(psi_look_T) - V_tot_M*cos(theta_look)*sin(psi_look) )/Range 
        LOS_g_dot                              = ( V_tot_T*sin(theta_look_T) - V_tot_M*sin(theta_look) )/Range 

        LOS_t_dot = np.clip(LOS_t_dot, -np.pi/2, np.pi/2)
        LOS_g_dot = np.clip(LOS_g_dot, -np.pi/2, np.pi/2)

        aq_M                                   = self.N_g*V_tot_M*LOS_g_dot # 유도탄 pitch 방향 (longitudinal) 가속 - PNG
        ar_M                                   = self.N_t*V_tot_M*LOS_t_dot # 유도탄 yaw 방향 (lateral) 가속 - PNG

        aq_M                                   = aq_M / self.env.control_gain
        ar_M                                   = ar_M / self.env.control_gain

        action                                 = np.zeros(2)
        
        action[0]                              = aq_M
        action[1]                              = ar_M
        # np.array([LOS_g_dot, LOS_t_dot])
        return action