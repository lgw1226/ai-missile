import os
import mmap
import time
import struct
import cv2 as cv
import subprocess
import numpy as np
import configparser
from numpy import pi
from numpy import linalg as LA
from gym_Missile.agents.common.data_augs import *

D2R = pi/180

class Render():
    def __init__(self, args):
        super(Render, self).__init__()
        self.args               = args

        # Interceptor
        self.shape              = (240, 320, 1)
        self.n                  = np.prod(self.shape)

        # Edit config.ini
        config = configparser.ConfigParser()
        config.optionxform=str
        config.read(os.getcwd()+'/Config.ini')
        config['Target']['Control'] = "./gym_Missile/render/binary_file/target_control_" + str(args.sim_idx)
        config['Target']['FLAME_SCALE'] = "100"
        config['Interceptor']['Control'] = "./gym_Missile/render/binary_file/interceptor_control_" + str(args.sim_idx)
        config['Interceptor']['CamBinary'] = "./gym_Missile/render/binary_file/interceptor_cam_binary_" + str(args.sim_idx)
        config['Interceptor']['Cam2'] = "./gym_Missile/render/binary_file/interceptor_cam2_" + str(args.sim_idx)
        config['Operation']['Info'] = "./gym_Missile/render/binary_file/operation_info_" + str(args.sim_idx)
        config['CamControl']['FOV'] = "90"
        config['CamControl']['FOV_MIN'] = "40"
        config['CamControl']['FOV_MAX'] = "80"
        config['CamControl']['DIST_MIN'] = "1"
        config['CamControl']['DIST_MAX'] = "10000"
        config['CamControl']['IMG_WIDTH'] = "320"
        config['CamControl']['IMG_HEIGHT'] = "240"
        config['MonoCam2Control']['FOV'] = "120"
        config['Explosion']['Threshold'] = "0"

        with open('Config.ini', 'w', encoding='utf-8') as configfile:
            config.write(configfile, space_around_delimiters=False)

        # Execute Simulator
        if self.args.os == 'window':
            path = os.getcwd() + "/gym_Missile/render/missile-win/Missile.exe"
            self.sim = subprocess.Popen(path)
        elif self.args.os == 'ubuntu':
            path = os.getcwd() + "/gym_Missile/render/missile-lin/missile.x86_64"
            self.sim = subprocess.Popen(path)

        # Binary File Read
        fd_binary_img          = os.open(os.getcwd() + "/gym_Missile/render/binary_file/interceptor_cam_binary_" + str(args.sim_idx), os.O_CREAT | os.O_RDWR)
        fd_binary_img_2        = os.open(os.getcwd() + "/gym_Missile/render/binary_file/interceptor_cam2_" + str(args.sim_idx), os.O_CREAT | os.O_RDWR)
        fd_operation_info      = os.open(os.getcwd() + "/gym_Missile/render/binary_file/operation_info_" + str(args.sim_idx), os.O_CREAT | os.O_RDWR)
        fd_target_control      = os.open(os.getcwd() + "/gym_Missile/render/binary_file/target_control_" + str(args.sim_idx), os.O_CREAT | os.O_RDWR)
        fd_interceptor_control = os.open(os.getcwd() + "/gym_Missile/render/binary_file/interceptor_control_" + str(args.sim_idx), os.O_CREAT | os.O_RDWR)

        # mmap Memory
        self.mm_operation_info      = mmap.mmap(fd_operation_info, 12)
        self.mm_binary_img          = mmap.mmap(fd_binary_img, self.n)
        self.mm_binary_img_2        = mmap.mmap(fd_binary_img_2, self.n)
        self.mm_target_control      = mmap.mmap(fd_target_control, 64)
        self.mm_interceptor_control = mmap.mmap(fd_interceptor_control, 80)
        
        self.sim_reset()
        self.select_mode()
        
    def select_mode(self):
        # Operation
        if self.args.phase != 'control':
            self.mode = 1 # Learn
            if self.args.phase == 'IL':
                self.sub_mode = 1 # IL
            elif self.args.phase == 'DRL':
                self.sub_mode = 2 # DRL
            else:
                self.sub_mode = 0 # Standby
            self.str_method = self.args.algo
        else:
            self.mode = 2 # Control
            self.sub_mode = 1 # None
            self.str_method = self.args.control_method
            
        if self.args.env == 'kinematics2d-v0':
            self.missile_dof   = 1 # 1: 3dof  2: 6dof
            self.missile_model = 1 # 1: kinematics 2: dynamics
        elif self.args.env == 'kinematics2d-v1' or self.args.env == 'kinematics2d-v2':
            self.missile_dof   = 1
            self.missile_model = 1
        elif self.args.env == 'kinematics3d-v0' or self.args.env == 'kinematics3d-v1':
            self.missile_dof   = 2
            self.missile_model = 1
        elif self.args.env == 'dynamics2d-v0':
            self.missile_dof   = 1
            self.missile_model = 2
        elif self.args.env == 'dynamics3d-v0':
            self.missile_dof   = 2
            self.missile_model = 2

    def sim_reset(self):
        for _ in range(100):
            self.operation(255, 0, 'Reset')
            time.sleep(0.01)
    
    def operation(self, mode, sub_mode, str_method):
        method = bytes(str_method, 'utf-8')
        operation_info_data = struct.pack('BB 10s', mode, sub_mode, method)
        self.mm_operation_info.seek(0)
        self.mm_operation_info.write(operation_info_data)
        self.mm_operation_info.flush()
    
    def missile(self, Mstates, Tstates, reward):
        if self.args.env == 'kinematics2d-v0':
            pos_M = Mstates[:2]; V_tot_M = 500
            x_M = pos_M[0]; y_M = 0.0; z_M = pos_M[1]; h_M = z_M
            phi_M = 0.0; theta_M = -Mstates[2]; psi_M = 0.0
            
            pos_T = Tstates[:2]; V_tot_T = 200
            x_T = pos_T[0]; y_T = 0.0; z_T = pos_T[1]; h_T = z_T
            phi_T = 0.0; theta_T = -Tstates[2]; psi_T = 0.0

        elif self.args.env == 'kinematics2d-v1' or self.args.env == 'kinematics2d-v2':
            pos_M = Mstates[:2]; V_tot_M = 1000
            x_M = pos_M[0]; y_M = 0.0; z_M = pos_M[1]; h_M = z_M
            phi_M = 0.0; theta_M = -Mstates[2]; psi_M = 0.0
            
            pos_T = Tstates[:2]; V_tot_T = 500
            x_T = pos_T[0]; y_T = 0.0; z_T = pos_T[1]; h_T = z_T
            phi_T = 0.0; theta_T = -Tstates[2]; psi_T = 0.0
        
        elif self.args.env == 'kinematics3d-v0' or self.args.env == 'kinematics3d-v1':
            pos_M = Mstates[:3]; vel_M = Mstates[3:6]; eul_M = Mstates[6:]
            x_M = pos_M[0]; y_M = -pos_M[1]; z_M = pos_M[2]; h_M = -z_M
            u_M = vel_M[0]; v_M = vel_M[1]; w_M = vel_M[2]; V_tot_M = LA.norm(vel_M)
            phi_M = eul_M[0]; theta_M = -eul_M[1]; psi_M = -eul_M[2]
            
            pos_T = Tstates[:3]; vel_T = Tstates[3:6]; eul_T = Tstates[6:]
            x_T = pos_T[0]; y_T = -pos_T[1]; z_T = pos_T[2]; h_T = -z_T
            u_T = vel_T[0]; v_T = vel_T[1]; w_T = vel_T[2]; V_tot_T = LA.norm(vel_T)
            phi_T = eul_T[0]; theta_T = -eul_T[1]; psi_T = -eul_T[2]
            
        elif self.args.env == 'dynamics2d-v0':
            pos_M = Mstates[:2]; V_tot_M = 500
            x_M  = pos_M[0]; y_M = 0.0; z_M = pos_M[1]; h_M = -z_M
            M_alpha = Mstates[2]; phi_M = 0; theta_M = -Mstates[3]; psi_M = 0
            
            pos_T = Tstates[:2]; V_tot_T = 300
            x_T = pos_T[0]; y_T = 0.0; z_T = pos_T[1]; h_T = -z_T
            phi_T = 0.0; theta_T = -Tstates[2]; psi_T = 0.0
            
        elif self.args.env == 'dynamics3d-v0':
            pass

        else:
            print('Error: environment')
            pass
            
        Range = LA.norm(pos_T - pos_M)
        
        interceptor_ctrl_data = struct.pack('dddddddBBdd', x_M, y_M, h_M, V_tot_M, phi_M/D2R, theta_M/D2R, psi_M/D2R, 
                                        self.missile_model, self.missile_dof, reward, Range)
        
        target_ctrl_data = struct.pack('dddddddBB', x_T, y_T, h_T, V_tot_T, phi_T/D2R, theta_T/D2R, psi_T/D2R, 
                                       self.missile_model, self.missile_dof)
    
        self.mm_interceptor_control.seek(0)
        self.mm_interceptor_control.write(interceptor_ctrl_data)
        self.mm_interceptor_control.flush()
        
        self.mm_target_control.seek(0)
        self.mm_target_control.write(target_ctrl_data)
        self.mm_target_control.flush()
        
    def thermal_img(self):
        self.mm_binary_img.seek(0)
        buf = self.mm_binary_img.read(self.n)
        img_ = np.frombuffer(buf, dtype=np.uint8).reshape(self.shape)
        img_ = img_.squeeze()
        # cv.imshow("Thermal Image", img_)
        # cv.waitKey(1) & 0xFF
        return img_
    
    def thermal_img_2(self):
        self.mm_binary_img_2.seek(0)
        buf = self.mm_binary_img_2.read(self.n)
        img_ = np.frombuffer(buf, dtype=np.uint8).reshape(self.shape)
        img_ = img_.squeeze()
        # cv.imshow("Thermal Image 2", img_)
        # cv.waitKey(1) & 0xFF
        return img_
