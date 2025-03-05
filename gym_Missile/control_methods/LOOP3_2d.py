import numpy as np
from math import atan2
from numpy import pi, exp, sin, cos
from numpy import linalg as LA
from scipy.linalg import solve_continuous_are

D2R = pi / 180
grav = 9.806

class Loop3_2d():
    
    def __init__(self, env): 
        super(Loop3_2d, self).__init__()
        self.env                = env
        self.V_M                = self.env.M_V
        self.V_T                = self.env.T_V     

        # Parameters        
        self.L_alpha_B          = 1190
        self.L_delta            = 80
        self.M_q                = -5 
        self.M_delta            = 160
        self.M_alpha_B          = -234 
        self.tau_q              = 0.02 
        self.int_q_M            = 0
        self.int_e_aM           = 0
        self.sat_angle          = 30 * D2R      
        
        # 3 Loop Autopilot
        A1_mat = np.array([[-(self.L_alpha_B+self.L_delta)/self.V_M, self.L_alpha_B+self.L_delta, 0],
                [0, 0, 1],
                [-(self.M_alpha_B+self.M_delta)/self.V_M, self.M_alpha_B+self.M_delta, self.M_q]])
        B1_mat = np.array([self.L_delta, 0, self.M_delta]).T
        Q1_mat = np.diag([1, 0, 0])
        R1_mat = 1e3
        
        
        [X_opt, self.K_opt, L_opt] = icare(A1_mat, B1_mat, Q1_mat, R1_mat, np.zeros(3,1), np.eye(3), np.zeros(3,3))
        
        
        Ac_mat = A1_mat - B1_mat*self.K_opt
        Bc_mat = -B1_mat*self.K_opt*np.array([1, 0, 0]).T
        Cc_mat = np.array([1, 0, 0])
        self.K_ss = LA.inv(Cc_mat/Ac_mat*Bc_mat)
        
        
    def control_input(self, Mstates, Tstates):        
        x_M             = Mstates[0]
        z_M             = Mstates[1]
        Pos_M           = Mstates[:2]
        alpha_M         = Mstates[2]
        theta_M         = Mstates[3]
        q_M             = Mstates[4]
        delta_q         = Mstates[5]
        gamma_M         = theta_M - alpha_M 

        x_T             = Tstates[0]
        z_T             = Tstates[1]        
        Pos_T           = Tstates[:2]
        gamma_T         = Tstates[2]
        
        # PNG 3D
        LOS             = atan2(-z_T + z_M, x_T - x_M)
        Range           = LA.norm(Pos_M - Pos_T)
        sigma_M         = gamma_M - LOS
        sigma_T         = gamma_T - LOS
        
        Range_dot       = self.V_T * cos(sigma_T) - self.V_M * cos(sigma_M)
        LOS_dot         = (self.V_T * sin(sigma_T) - self.V_M * sin(sigma_M)) / Range

        Lift_mass       = self.L_alpha_B*max(min(alpha_M, self.sat_angle),-self.sat_angle) + self.L_delta*max(min(alpha_M+delta_q,self.sat_angle),-self.sat_angle)
        M_I             = self.M_alpha_B*max(min(alpha_M, self.sat_angle),-self.sat_angle) + self.M_q*q_M + self.M_delta*max(min(alpha_M+delta_q, self.sat_angle),-self.sat_angle)
        a_M             = Lift_mass        

        N_PN = 5
        a_M_c = N_PN*self.V_M*LOS_dot
    
        # 3 Loop
        e_aM = a_M - self.K_ss*a_M_c
        delta_q_c = -self.K_opt(1,1)*self.int_e_aM - self.K_opt(1,2)*self.int_q_M - self.K_opt(1,3)*q_M

        delta_q_c       = max(min(delta_q_c,self.sat_angle),-self.sat_angle)
        
        delta_q_c       = delta_q_c / self.sat_angle

        return delta_q_c