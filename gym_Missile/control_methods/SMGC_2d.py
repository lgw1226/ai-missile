import numpy as np
from math import atan2
from numpy import pi, exp, sin, cos
from numpy import linalg as LA
from scipy.linalg import expm

D2R = pi / 180
grav = 9.806


def clipped_exp(x):
    return np.exp(np.clip(x, -100, 100))


class SMGC_2d():
    def __init__(self, env, gain: int = 200, **kwargs):
        super(SMGC_2d, self).__init__()
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
        self.tau_T              = 0.1
        self.sat_angle          = 30 * D2R        
        
        # init Variables
        self.gamma_M_0          = 0
        self.gamma_T_0          = 0
        self.LOS_0              = 0

        self.gain = gain
        
        
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
        
        LOS             = atan2(-z_T + z_M, x_T - x_M)
        # LOS             = LOS * np.random.normal(loc = 1.0, scale = 0.001)
        Range           = LA.norm(Pos_M - Pos_T)
        sigma_M         = gamma_M - LOS
        sigma_T         = gamma_T - LOS
        
        Range_dot       = self.V_T * cos(sigma_T) - self.V_M * cos(sigma_M)
        LOS_dot         = (self.V_T * sin(sigma_T) - self.V_M * sin(sigma_M)) / Range
        # LOS_dot         = LOS_dot * np.random.normal(loc = 1.0, scale = 0.001)
        V_r             = Range_dot
        V_lambda        = Range * LOS_dot
        
        Lift_mass       = self.L_alpha_B*max(min(alpha_M, self.sat_angle),-self.sat_angle) + self.L_delta*max(min(alpha_M+delta_q,self.sat_angle),-self.sat_angle)
        M_I             = self.M_alpha_B*max(min(alpha_M, self.sat_angle),-self.sat_angle) + self.M_q*q_M + self.M_delta*max(min(alpha_M+delta_q, self.sat_angle),-self.sat_angle)
        a_M             = Lift_mass
        
        a_T             = -5 * grav
        V_r_dot         = np.power(V_lambda,2)/Range + a_M*sin(sigma_M) - a_T*sin(sigma_T)
        
        # State Space
        s_GC_gain       = self.gain
        A_G11           = np.array([[0, 1, 0], 
                                    [0, 0, 1],
                                    [0, 0, -1/self.tau_T]
                                    ])
                
        if self.env.count == 0:
            self.gamma_M_0 = gamma_M
            self.gamma_T_0 = gamma_T
            self.LOS_0     = LOS
        else:
            pass
        
        a_TN            = a_T * cos(self.gamma_T_0 - self.LOS_0)
        L_alpha         = self.L_alpha_B + self.L_delta
        M_alpha         = self.M_alpha_B + self.M_delta
        
        A_M             = np.array([[-L_alpha/self.V_M, 1, -self.L_delta/self.V_M],
                                    [M_alpha, self.M_q, self.M_delta],
                                    [0, 0, -1/self.tau_q]
                                    ])
        B_M             = np.array([[0],
                                    [0],
                                    [1/self.tau_q]
                                    ])
        C_M             = np.array([L_alpha, 0, self.L_delta]) * cos(self.gamma_M_0 - self.LOS_0)
        
        A_12            = np.zeros([3,3])
        A_12[1]         = -C_M
        
        A_GC            = np.zeros([6,6])
        A_GC[0:3,0:3]   = A_G11
        A_GC[0:3,3:6]   = A_12
        A_GC[3:,3:]     = A_M
        
        B_GC            = np.zeros([6,1])
        B_GC[5]         = 1/self.tau_q
        
        C_GC            = np.zeros(6)
        C_GC[0]         = 1
        
        # Control
        tgo_est         = - Range / Range_dot
        A_GC_tgo        = np.dot(A_GC, tgo_est)
        # A_GC_tgo = np.clip(A_GC_tgo, -100, 100)
        
        Phi_GC          = expm(A_GC_tgo)
        
        Phi_GC_final    = Phi_GC[0][5]
        x_GC            = np.array([0, 0, 0, alpha_M, q_M, delta_q]).T
        
        ZEM_GC          = -Range_dot*np.power(tgo_est,2)*LOS_dot + a_TN*np.power(self.tau_T,2)*(exp(-tgo_est/self.tau_T) + tgo_est/self.tau_T - 1) + np.dot(np.dot(C_GC,Phi_GC),x_GC)
        delta_q_eq      = -(Range*LOS_dot + a_TN*self.tau_T*(1-exp(-tgo_est/self.tau_T)) + np.dot(np.dot(np.dot(C_GC,Phi_GC),A_GC),x_GC))*(V_r_dot*Range*self.tau_q)/(np.power(V_r,2)*Phi_GC_final)
        delta_q_c       = delta_q_eq - s_GC_gain * max(min(ZEM_GC, 1.5),-1.5) * self.tau_q/Phi_GC_final

        delta_q_c       = max(min(delta_q_c, self.sat_angle), -self.sat_angle)
        
        delta_q_c       = delta_q_c / self.sat_angle
        if delta_q_c == np.NaN or delta_q_c == np.Inf:
            delta_q_c = 0
        return delta_q_c