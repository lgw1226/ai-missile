import numpy as np
from math import pi, atan2
from numpy import pi, cos, sin
from numpy import linalg as LA

D2R = pi / 180

class PNG_2d:
    def __init__(self, env, gain: int = 5):
        super(PNG_2d, self).__init__()
        self.env            = env
        self.N_PN           = gain

    def control_input(self, Mstates, Tstates):
        M_V   = self.env.M_V
        M_Pos = Mstates[:2]
        M_gam = Mstates[2]
        
        T_V   = self.env.T_V
        T_Pos = Tstates[:2]
        T_gam = Tstates[2]

        Range = LA.norm(T_Pos-M_Pos)
        
        LOS   = atan2(T_Pos[1]-M_Pos[1], T_Pos[0]-M_Pos[0])
        LOS_dot = (T_V*sin(T_gam-LOS) - M_V*sin(M_gam-LOS)) / Range
        LOS_dot = np.clip(LOS_dot, -np.pi/2, np.pi/2)
        
        a_M_c   = self.N_PN * M_V * LOS_dot
        a_M_c = max(min(a_M_c, self.env.control_gain), -self.env.control_gain)

        a_M_c   = a_M_c / self.env.control_gain
              
        return a_M_c