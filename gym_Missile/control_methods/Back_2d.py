import numpy as np
from math import atan2
from numpy import pi, exp, sin, cos
from numpy import linalg as LA

D2R = pi / 180
grav = 9.806

class Back_2d():
    
    def __init__(self, env): 
        super(Back_2d, self).__init__()
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
        self.sat_angle          = 30 * D2R      
        
        
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
        # LOS             = np.random.normal(loc = 1.0, scale = 0.01, size = 1) * LOS
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
    
        # Backstepping
        k_1 = 10
        k_2 = 10
        
        e_aM = a_M - a_M_c
        q_Mc = a_M/self.V_M - 1/self.L_alpha_B*k_1*e_aM
        e_q = q_M - q_Mc
        delta_q_c = -1/self.M_delta*( (self.M_alpha_B+self.M_delta)*alpha_M + self.M_q*q_M + k_2*e_q )
        
        delta_q_c       = max(min(delta_q_c,self.sat_angle),-self.sat_angle)
        
        delta_q_c       = delta_q_c / self.sat_angle

        return delta_q_c