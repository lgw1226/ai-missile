import os
import gym
import time
import argparse
import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from math import pi, atan2, cos, sin, tan, asin, acos, sqrt
from numpy import linalg as LA
from ambiance import Atmosphere

D2R = pi / 180

class Dynamics:
    def __init__(self, args):
        self.M_states    = np.zeros(args.state_dim)
        self.T_states    = np.zeros(args.state_dim)
        self.dt          = 1/(args.step_hz*np.ones(1))
        self.grav        = 9.806
        self.args        = args
        
        '''
        Missile : N, E, D, V, alpha, beta, phi, theta, psi, p, q, r, del_e, del_r, T_y, T_z
        Target  : N, E, D, V, alpha, beta, phi, theta, psi
        '''    

    def reset(self, episode):
        self.M_states    = np.zeros(self.args.state_dim)
        self.T_states    = np.zeros(self.args.state_dim)
        
        self.M_states[0] = self.args.m_N 
        self.M_states[1] = self.args.m_E
        self.M_states[2] = self.args.m_D
        self.M_states[3] = self.args.m_u
        self.M_states[4] = 0
        self.M_states[5] = 0

        self.T_states[0] = self.args.t_N 
        self.T_states[1] = self.args.t_E
        self.T_states[2] = self.args.t_D
        self.T_states[3] = self.args.t_u
        
        
        # Parameters, Coefficient
        _atmo_trim                              = Atmosphere(22000) #22km
        T_trim                                  = float(_atmo_trim.temperature); 
        a_trim                                  = float(_atmo_trim.speed_of_sound)
        P_trim                                  = float(_atmo_trim.pressure) 
        rho_trim                                = float(_atmo_trim.density)
        Mach_trim                               = 3.7
        V_M_trim                                = Mach_trim*a_trim
        q_dyn_trim                              = 1/2*rho_trim*(V_M_trim**2)
        
        self.I_y                                = 150
        self.I_z                                = 150
        self.mass_M                             = 150
        self.l_T                                = 1.1
        self.tau_a                              = 0.005
        self.tau_f                              = 0.001
        
        c_1 = -0.1; c_2 = -4.1; c_3 = -24.3; c_4 = -0.11; c_5 = -0.02; c_6 = 0.0073; c_7 = 5.99e-6
        c_1_pr = c_1; c_2_pr = c_2; c_3_pr = -c_3; c_4_pr = c_4; c_5_pr = c_5; c_6_pr = c_6
        
        self.S_L_Cmq                            = c_1*self.I_y/q_dyn_trim
        self.S_L_Cma                            = c_2*self.I_y/q_dyn_trim
        self.S_L_Cmde                           = c_3*self.I_y/q_dyn_trim
        self.S_Ca                               = c_4*self.mass_M*V_M_trim/q_dyn_trim
        self.S_Cde                              = c_5*self.mass_M*V_M_trim/q_dyn_trim
        self.S_L_Cmr                            = c_1_pr*self.I_z/q_dyn_trim
        self.S_L_Cmb                            = c_2_pr*self.I_z/q_dyn_trim
        self.S_L_Cmdr                           = c_3_pr*self.I_z/q_dyn_trim
        self.S_Cb                               = c_4_pr*self.mass_M*V_M_trim/q_dyn_trim
        self.S_Cdr                              = c_5_pr*self.mass_M*V_M_trim/q_dyn_trim
        self.fin_max                            = 45*D2R; self.T_max = 5e3
        

    def get_diff(self, action, Mstates, Tstates):

        pos_M = Mstates[:3]; vel_M = Mstates[3:6]; eul_M = Mstates[6:9]; pqr_M = Mstates[9:12]; del_M = Mstates[12:16]
        x_M = pos_M[0]; y_M = pos_M[1]; z_M = pos_M[2]; h_M = -float(z_M)
        V_M = vel_M[0]; alpha_M = vel_M[1]; beta_M= vel_M[2]
        u_M = V_M*cos(alpha_M)*cos(beta_M); v_M = V_M*sin(beta_M); w_M = V_M*sin(alpha_M)*cos(beta_M)
        phi_M = eul_M[0]; theta_M = eul_M[1]; psi_M = eul_M[2]
        p_M = pqr_M[0]; q_M = pqr_M[1]; r_M = pqr_M[2]
        del_e = del_M[0]; del_r = del_M[1]; T_y = del_M[2]; T_z = del_M[3]
        
        pos_T = Tstates[:3]; vel_T = Tstates[3:6]; eul_T = Tstates[6:]
        x_T = pos_T[0]; y_T = pos_T[1]; z_T = pos_T[2]; h_T = -z_T
        V_T = vel_T[0]; alpha_T = vel_T[1]; beta_T = vel_T[2]
        u_T = V_T*cos(alpha_T)*cos(beta_T); v_T = V_T*sin(beta_T); w_T = V_T*sin(alpha_T)*cos(beta_T)
        phi_T = eul_T[0]; theta_T = eul_T[1]; psi_T = eul_T[2]
  
        del_ec = action[0]
        del_rc = action[1]
        T_yc = action[2]
        T_zc = action[3]
        
        _atmo                                   = Atmosphere(h_M)
        rho_dst                                 = float(_atmo.density)
        q_dyn                                   = 1/2*rho_dst*(V_M**2)
        F_y                                     = q_dyn*(self.S_Cb*beta_M + self.S_Cdr*del_r)
        F_z                                     = q_dyn*(self.S_Ca*alpha_M + self.S_Cde*del_e)
        M_a                                     = q_dyn*(self.S_L_Cma*alpha_M + self.S_L_Cmde*del_e + self.S_L_Cmq*q_M)
        N_a                                     = q_dyn*(self.S_L_Cmb*beta_M + self.S_L_Cmdr*del_r + self.S_L_Cmr*r_M)       

        aq_T                                    = self.grav
        ar_T                                    = self.grav
        q_T                                     = aq_T/V_T
        r_T                                     = ar_T/V_T
             
        x_M_dot = u_M*cos(theta_M)*cos(psi_M) + v_M*(-cos(phi_M)*sin(psi_M)+sin(phi_M)*sin(theta_M)*cos(psi_M)) + w_M*(sin(phi_M)*sin(psi_M)+cos(phi_M)*sin(theta_M)*cos(psi_M))
        y_M_dot = u_M*cos(theta_M)*sin(psi_M) + v_M*(cos(phi_M)*cos(psi_M)+sin(phi_M)*sin(theta_M)*sin(psi_M)) + w_M*(-sin(phi_M)*cos(psi_M)+cos(phi_M)*sin(theta_M)*sin(psi_M))
        z_M_dot = -u_M*sin(theta_M) + v_M*sin(phi_M)*cos(theta_M) + w_M*cos(phi_M)*cos(theta_M); h_M_dot = -z_M_dot
        V_M_dot = 0
        alpha_M_dot = q_M - r_M*sin(alpha_M)*tan(beta_M) + (F_z + T_z)*cos(alpha_M)/(self.mass_M*V_M*cos(beta_M))
        beta_M_dot = -r_M*cos(alpha_M) + ( (F_y + T_y)*cos(beta_M) + (F_z + T_z)*sin(alpha_M)*sin(beta_M) )/(self.mass_M*V_M)
        phi_M_dot = 0
        theta_M_dot = q_M*cos(phi_M) - r_M*sin(phi_M)
        psi_M_dot = q_M*sin(phi_M)/cos(theta_M) + r_M*cos(phi_M)/cos(theta_M)
        p_M_dot = 0
        q_M_dot = (M_a - T_z*self.l_T)/self.I_z
        r_M_dot = (N_a + T_y*self.l_T)/self.I_y
        del_e_dot = (del_ec - del_e)/self.tau_a
        del_r_dot = (del_rc - del_r)/self.tau_a
        T_y_dot = (T_yc - T_y)/self.tau_f
        T_z_dot = (T_zc - T_z)/self.tau_f

        x_T_dot = u_T*cos(theta_T)*cos(psi_T) + v_T*(-cos(phi_T)*sin(psi_T)+sin(phi_T)*sin(theta_T)*cos(psi_T)) + w_T*(sin(phi_T)*sin(psi_T)+cos(phi_T)*sin(theta_T)*cos(psi_T));
        y_T_dot = u_T*cos(theta_T)*sin(psi_T) + v_T*(cos(phi_T)*cos(psi_T)+sin(phi_T)*sin(theta_T)*sin(psi_T)) + w_T*(-sin(phi_T)*cos(psi_T)+cos(phi_T)*sin(theta_T)*sin(psi_T));
        z_T_dot = -u_T*sin(theta_T) + v_T*sin(phi_T)*cos(theta_T) + w_T*cos(phi_T)*cos(theta_T); h_T_dot = -z_T_dot
        V_T_dot = 0
        alpha_T_dot = 0
        beta_T_dot = 0
        phi_T_dot = 0
        theta_T_dot = q_T*cos(phi_T) - r_T*sin(phi_T)
        psi_T_dot = q_T*sin(phi_T)/cos(theta_T) + r_T*cos(phi_T)/cos(theta_T)


        _Mstates       = np.zeros(self.args.state_dim)
        _Tstates       = np.zeros(self.args.state_dim)

        _Mstates[0]    = x_M_dot 
        _Mstates[1]    = y_M_dot 
        _Mstates[2]    = z_M_dot 
        _Mstates[3]    = V_M_dot
        _Mstates[4]    = alpha_M_dot
        _Mstates[5]    = beta_M_dot
        _Mstates[6]    = phi_M_dot
        _Mstates[7]    = theta_M_dot
        _Mstates[8]    = psi_M_dot
        _Mstates[9]    = p_M_dot
        _Mstates[10]    = q_M_dot
        _Mstates[11]    = r_M_dot
        _Mstates[12]    = del_e_dot
        _Mstates[13]    = del_r_dot
        _Mstates[14]    = T_y_dot
        _Mstates[15]    = T_z_dot

        _Tstates[0]    = x_T_dot 
        _Tstates[1]    = y_T_dot 
        _Tstates[2]    = z_T_dot 
        _Tstates[3]    = V_T_dot 
        _Tstates[4]    = alpha_T_dot 
        _Tstates[5]    = beta_T_dot 
        _Tstates[6]    = phi_T_dot 
        _Tstates[7]    = theta_T_dot 
        _Tstates[8]    = psi_T_dot 
        _Tstates[9]    = 0
        _Tstates[10]    = 0
        _Tstates[11]    = 0
        _Tstates[12]    = 0
        _Tstates[13]    = 0
        _Tstates[14]    = 0
        _Tstates[15]    = 0
        

        return _Mstates, _Tstates
    
    def get_input(self, Mstates, Tstates, initial_range, initial_los_lat, initial_los_lat_dot, initial_los_lon, initial_los_lon_dot):
        pos_M = Mstates[:3]; vel_M = Mstates[3:6]; eul_M = Mstates[6:9]; pqr_M = Mstates[9:12]; del_M = Mstates[12:16]
        x_M = pos_M[0]; y_M = pos_M[1]; z_M = pos_M[2]; h_M = -float(z_M);
        V_M = vel_M[0]; alpha_M = vel_M[1]; beta_M= vel_M[2]
        u_M = V_M*cos(alpha_M)*cos(beta_M); v_M = V_M*sin(beta_M); w_M = V_M*sin(alpha_M)*cos(beta_M)
        phi_M = eul_M[0]; theta_M = eul_M[1]; psi_M = eul_M[2]
        p_M = pqr_M[0]; q_M = pqr_M[1]; r_M = pqr_M[2]
        del_e = del_M[0]; del_r = del_M[1]; T_y = del_M[2]; T_z = del_M[3]
        
        pos_T = Tstates[:3]; vel_T = Tstates[3:6]; eul_T = Tstates[6:]
        x_T = pos_T[0]; y_T = pos_T[1]; z_T = pos_T[2]; h_T = -z_T
        V_T = vel_T[0]; alpha_T = vel_T[1]; beta_T = vel_T[2]
        u_T = V_T*cos(alpha_T)*cos(beta_T); v_T = V_T*sin(beta_T); w_T = V_T*sin(alpha_T)*cos(beta_T)
        phi_T = eul_T[0]; theta_T = eul_T[1]; psi_T = eul_T[2];

        N_PN                                   = 5
        Range                                  = LA.norm(pos_T - pos_M)
        LOS_t                                  = atan2(y_T-y_M,x_T-x_M)
        LOS_g                                  = asin((z_T-z_M)/Range)

        rotm_B2W                               = self.eul2rotm(-beta_M, alpha_M, 0)
        rotm_B2I                               = self.eul2rotm(psi_M, theta_M, phi_M)
        rotm_L2I                               = self.eul2rotm(LOS_t, -LOS_g, 0)
        rotm_L2W                               = np.dot(np.transpose(rotm_B2I),rotm_L2I)
        eul_look                               = self.rotm2eul(np.transpose(rotm_L2W))

        phi_look                               = eul_look[2]
        theta_look                             = eul_look[1] 
        psi_look                               = eul_look[0]

        rotm_T2I                               = self.eul2rotm(psi_T, theta_T, phi_T)
        rotm_L2I                               = self.eul2rotm(LOS_t, -LOS_g, 0)
        rotm_L2T                               = np.dot(np.transpose(rotm_T2I),rotm_L2I)
        eul_look_T                             = self.rotm2eul(np.transpose(rotm_L2T))
        phi_look_T                             = eul_look_T[2]
        theta_look_T                           = eul_look_T[1]
        psi_look_T                             = eul_look_T[0]
        
        Range_dot                              = ( V_T*cos(theta_look_T)*cos(psi_look_T) - V_M*cos(theta_look)*cos(psi_look) ) 
        LOS_t_dot                              = ( V_T*cos(theta_look_T)*sin(psi_look_T) - V_M*cos(theta_look)*sin(psi_look) )/Range 
        LOS_g_dot                              = ( V_T*sin(theta_look_T) - V_M*sin(theta_look) )/Range 

        if self.args.obs_dim == 4:
            NN_input    = torch.zeros(self.args.obs_dim)
            # NN_input[0] = -(psi_look - initial_psi_look)*(psi_look - initial_psi_look)
            # NN_input[1] = -LOS_t_dot*LOS_t_dot
            # NN_input[2] = -(theta_look - initial_theta_look)*(theta_look - initial_theta_look)
            # NN_input[3] = -LOS_g_dot*LOS_g_dot
            # NN_input[0] = psi_look - initial_psi_look
            # NN_input[1] = LOS_t_dot
            # NN_input[2] = theta_look - initial_theta_look
            # NN_input[3] = LOS_g_dot
            # NN_input[0] = LOS_t - psi_M
            # NN_input[1] = LOS_t_dot
            # NN_input[2] = LOS_g - theta_M
            # NN_input[3] = LOS_g_dot            

            NN_input[0] = LOS_t / initial_los_lat
            NN_input[1] = LOS_t_dot / initial_los_lat_dot
            NN_input[2] = LOS_g / initial_los_lon
            NN_input[3] = LOS_g_dot / initial_los_lon_dot
                  
        elif self.args.obs_dim == 5:
            NN_input    = torch.zeros(self.args.obs_dim)
            # NN_input[0] = np.exp(-(psi_look - initial_psi_look)*(psi_look - initial_psi_look))
            # NN_input[1] = np.exp(-LOS_t_dot*LOS_t_dot)
            # NN_input[2] = np.exp(-(theta_look - initial_theta_look)*(theta_look - initial_theta_look))
            # NN_input[3] = np.exp(-LOS_g_dot*LOS_g_dot)
            # NN_input[0] = psi_look - initial_psi_look
            # NN_input[1] = LOS_t_dot
            # NN_input[2] = theta_look - initial_theta_look
            # NN_input[3] = LOS_g_dot
            # NN_input[0] = LOS_t - psi_M
            # NN_input[1] = LOS_t_dot
            # NN_input[2] = LOS_g - theta_M
            # NN_input[3] = LOS_g_dot
            # NN_input[4] = 1 - np.exp(-Range)

            NN_input[0] = LOS_t / initial_los_lat
            NN_input[1] = LOS_t_dot / initial_los_lat_dot
            NN_input[2] = LOS_g / initial_los_lon
            NN_input[3] = LOS_g_dot / initial_los_lon_dot
            NN_input[4] = Range / initial_range

        NN_input = torch.unsqueeze(NN_input, 0)      
        return NN_input    
    
    
    
    def get_reward(self, Mstates, Tstates, initial_range, initial_los_lat, initial_los_lat_dot, initial_los_lon, initial_los_lon_dot, initial_zem1, initial_zem2, action_):
        pos_M = Mstates[:3]; vel_M = Mstates[3:6]; eul_M = Mstates[6:9]; pqr_M = Mstates[9:12]; del_M = Mstates[12:16]
        x_M = pos_M[0]; y_M = pos_M[1]; z_M = pos_M[2]; h_M = -float(z_M);
        V_M = vel_M[0]; alpha_M = vel_M[1]; beta_M= vel_M[2]
        u_M = V_M*cos(alpha_M)*cos(beta_M); v_M = V_M*sin(beta_M); w_M = V_M*sin(alpha_M)*cos(beta_M)
        phi_M = eul_M[0]; theta_M = eul_M[1]; psi_M = eul_M[2]
        p_M = pqr_M[0]; q_M = pqr_M[1]; r_M = pqr_M[2]
        del_e = del_M[0]; del_r = del_M[1]; T_y = del_M[2]; T_z = del_M[3]
        
        pos_T = Tstates[:3]; vel_T = Tstates[3:6]; eul_T = Tstates[6:]
        x_T = pos_T[0]; y_T = pos_T[1]; z_T = pos_T[2]; h_T = -z_T
        V_T = vel_T[0]; alpha_T = vel_T[1]; beta_T = vel_T[2]
        u_T = V_T*cos(alpha_T)*cos(beta_T); v_T = V_T*sin(beta_T); w_T = V_T*sin(alpha_T)*cos(beta_T)
        phi_T = eul_T[0]; theta_T = eul_T[1]; psi_T = eul_T[2];

        N_PN                                   = 5
        Range                                  = LA.norm(pos_T - pos_M)
        LOS_t                                  = atan2(y_T-y_M,x_T-x_M)
        LOS_g                                  = asin((z_T-z_M)/Range)

        rotm_B2W                               = self.eul2rotm(-beta_M, alpha_M, 0)
        rotm_B2I                               = self.eul2rotm(psi_M, theta_M, phi_M)
        rotm_L2I                               = self.eul2rotm(LOS_t, -LOS_g, 0)
        rotm_L2W                               = np.dot(np.transpose(rotm_B2I),rotm_L2I)
        eul_look                               = self.rotm2eul(np.transpose(rotm_L2W))

        phi_look                               = eul_look[2]
        theta_look                             = eul_look[1] 
        psi_look                               = eul_look[0]

        rotm_T2I                               = self.eul2rotm(psi_T, theta_T, phi_T)
        rotm_L2I                               = self.eul2rotm(LOS_t, -LOS_g, 0)
        rotm_L2T                               = np.dot(np.transpose(rotm_T2I),rotm_L2I)
        eul_look_T                             = self.rotm2eul(np.transpose(rotm_L2T))
        phi_look_T                             = eul_look_T[2]
        theta_look_T                           = eul_look_T[1]
        psi_look_T                             = eul_look_T[0]
        
        Range_dot                              = ( V_T*cos(theta_look_T)*cos(psi_look_T) - V_M*cos(theta_look)*cos(psi_look) ) 
        LOS_t_dot                              = ( V_T*cos(theta_look_T)*sin(psi_look_T) - V_M*cos(theta_look)*sin(psi_look) )/Range 
        LOS_g_dot                              = ( V_T*sin(theta_look_T) - V_M*sin(theta_look) )/Range 

        Vr        = Range_dot
        Vl_t      = Range * LOS_t_dot
        Vl_g      = Range * LOS_g_dot
        V_tot_    = sqrt(Vr**2+Vl_t**2+Vl_g**2)
        Vl_       = sqrt(Vl_g**2+Vl_t**2)
        zem       = Range * Vl_ / V_tot_

        if Vr > 0:
            rvr1 = -2
        else:
            rvr1 = 0 

        if Range/initial_range > 1:
            rvr2 = -2
        else:
            rvr2 = 0 

        action_reward = -abs(action_[0])/self.args.act_limit-abs(action_[1])/self.args.act_limit
        rz        = - (zem/abs(initial_zem1))**2 

        # reward    = (rvr1 + rvr2 + rz)*torch.ones(1).squeeze(0)
        reward    = (rvr1 + rvr2*1+ rz + action_reward)*torch.ones(1).squeeze(0)

        return reward, LOS_t_dot, LOS_g_dot 
    

    def get_done_chk_value(self, Mstates, Tstates):
        
        pos_M = Mstates[:3]; vel_M = Mstates[3:6]; eul_M = Mstates[6:9]; pqr_M = Mstates[9:12]; del_M = Mstates[12:16]
        x_M = pos_M[0]; y_M = pos_M[1]; z_M = pos_M[2]; h_M = -float(z_M);
        V_M = vel_M[0]; alpha_M = vel_M[1]; beta_M= vel_M[2]
        u_M = V_M*cos(alpha_M)*cos(beta_M); v_M = V_M*sin(beta_M); w_M = V_M*sin(alpha_M)*cos(beta_M)
        phi_M = eul_M[0]; theta_M = eul_M[1]; psi_M = eul_M[2]
        p_M = pqr_M[0]; q_M = pqr_M[1]; r_M = pqr_M[2]
        del_e = del_M[0]; del_r = del_M[1]; T_y = del_M[2]; T_z = del_M[3]
        
        pos_T = Tstates[:3]; vel_T = Tstates[3:6]; eul_T = Tstates[6:]
        x_T = pos_T[0]; y_T = pos_T[1]; z_T = pos_T[2]; h_T = -z_T
        V_T = vel_T[0]; alpha_T = vel_T[1]; beta_T = vel_T[2]
        u_T = V_T*cos(alpha_T)*cos(beta_T); v_T = V_T*sin(beta_T); w_T = V_T*sin(alpha_T)*cos(beta_T)
        phi_T = eul_T[0]; theta_T = eul_T[1]; psi_T = eul_T[2];

        N_PN                                   = 5
        Range                                  = LA.norm(pos_T - pos_M)
        LOS_t                                  = atan2(y_T-y_M,x_T-x_M)
        LOS_g                                  = asin((z_T-z_M)/Range)

        rotm_B2W                               = self.eul2rotm(-beta_M, alpha_M, 0)
        rotm_B2I                               = self.eul2rotm(psi_M, theta_M, phi_M)
        rotm_L2I                               = self.eul2rotm(LOS_t, -LOS_g, 0)
        rotm_L2W                               = np.dot(np.transpose(rotm_B2I),rotm_L2I)
        eul_look                               = self.rotm2eul(np.transpose(rotm_L2W))

        phi_look                               = eul_look[2]
        theta_look                             = eul_look[1] 
        psi_look                               = eul_look[0]

        rotm_T2I                               = self.eul2rotm(psi_T, theta_T, phi_T)
        rotm_L2I                               = self.eul2rotm(LOS_t, -LOS_g, 0)
        rotm_L2T                               = np.dot(np.transpose(rotm_T2I),rotm_L2I)
        eul_look_T                             = self.rotm2eul(np.transpose(rotm_L2T))
        phi_look_T                             = eul_look_T[2]
        theta_look_T                           = eul_look_T[1]
        psi_look_T                             = eul_look_T[0]
        
        Range_dot                              = ( V_T*cos(theta_look_T)*cos(psi_look_T) - V_M*cos(theta_look)*cos(psi_look) ) 
        LOS_t_dot                              = ( V_T*cos(theta_look_T)*sin(psi_look_T) - V_M*cos(theta_look)*sin(psi_look) )/Range 
        LOS_g_dot                              = ( V_T*sin(theta_look_T) - V_M*sin(theta_look) )/Range 
        LOS_y_dot                              = -LOS_g_dot
        LOS_z_dot                              = LOS_t_dot

        a_gc                                   = -N_PN*V_M*LOS_y_dot*cos(psi_look)
        a_tc                                   = N_PN*V_M*LOS_z_dot*cos(theta_look) - N_PN*V_M*LOS_y_dot*sin(theta_look)*sin(psi_look)
        
        a_zc                                   = -a_gc*cos(alpha_M) - a_tc*sin(alpha_M)*sin(beta_M)
        a_yc                                   = a_tc*cos(beta_M)
        
        Vr        = Range_dot
        Vl_t      = Range * LOS_t_dot
        Vl_g      = Range * LOS_g_dot
        V_tot_    = sqrt(Vr**2+Vl_t**2+Vl_g**2)
        Vl_       = sqrt(Vl_g**2+Vl_t**2)
        zem       = Range * Vl_ / V_tot_

        done_chk_value                                  = np.zeros(9)
        done_chk_value[0]                               = LOS_t
        done_chk_value[1]                               = psi_M
        done_chk_value[2]                               = LOS_g
        done_chk_value[3]                               = theta_M
        done_chk_value[4]                               = Range
        done_chk_value[5]                               = zem
        done_chk_value[6]                               = LOS_t_dot
        done_chk_value[7]                               = LOS_g_dot
        done_chk_value[8]                               = zem
        
        return done_chk_value

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
    def isRotationMatrix(R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    def rotm2eul(self, R):
    

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
    
    def anglelimit(self, angle):
        if abs(angle) >= pi:
            if angle >= 0:
                angle = angle - 2*pi;
            else:
                angle = angle + 2*pi 
        return angle