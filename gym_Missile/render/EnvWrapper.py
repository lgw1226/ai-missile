import os
import gym
import time
import cv2 as cv
import gym.spaces
from collections import deque
from numpy import linalg as LA
from gym_Missile.render.Render import Render
from gym_Missile.agents.common.data_augs import *
from argparse import Namespace


SIM_DELAY = 0.2
STEP_DELAY = 0.002


class EnvWrapper(gym.Wrapper, Render):
    def __init__(self, env, args):
        super(EnvWrapper, self).__init__(env)
        if args.render:
            time.sleep(SIM_DELAY) # wating for simulator loading
        self.env                = env
        self.args               = args
        self.image_count        = 0
        
    def reset(self):
        obs, info = self.env.reset()
        if self.args.render:
            self.sim_reset()
            self.image_count = 0
        
        return obs, info
        
    def render(self):
        time.sleep(STEP_DELAY) # Simulator time delay
        self.operation(self.mode, self.sub_mode, self.str_method)
        self.missile(self.env.M_states, self.env.T_states, self.env.reward)
        _img = self.thermal_img()
        _img_2 = self.thermal_img_2()
        if self.args.save_video:
            _img_path = self.args.video_path + "/" + str(self.env.episode)
            try:
                if not os.path.exists(_img_path):
                    os.makedirs(_img_path)
            except OSError:
                print("Error: Cannot create the directory {}".format(_img_path))

            cv.imwrite(_img_path + "/cam1_" + str(self.image_count) + ".png", _img)
            cv.imwrite(_img_path + "/cam2_" + str(self.image_count) + ".png", _img_2)
            self.image_count += 1


class ImageEnvWrapper(gym.Wrapper, Render):
    def __init__(self, env, args):
        super(ImageEnvWrapper, self).__init__(env)
        if args.render:
            Render.__init__(self, args)
            time.sleep(SIM_DELAY) # wating for simulator loading
        self.env                = env
        self.args               = args
        self.observation_space  = gym.spaces.Box(low=0, high=255,
                                            shape=(1, 86, 86), dtype=np.uint8)

    def reset(self):
        _, info = self.env.reset()
        self.sim_reset()
        img_obs = self.img_processing(self.thermal_img())
        img_obs = np.expand_dims(img_obs, axis=0)
        
        return img_obs, info
    
    def img_processing(self, img):
        img = cv.resize(img, (86,86), interpolation=cv.INTER_AREA)
        # img = random_cutout(img)
        # img = random_jiterring(img)
        
        return img
        
    def render(self):
        pass

    def step(self, action):
        time.sleep(STEP_DELAY) # Simulator time delay
        self.operation(self.mode, self.sub_mode, self.str_method)
        self.missile(self.env.M_states, self.env.T_states, self.env.reward)
        obs, reward, done, info = self.env.step(action)
        img_obs = self.img_processing(self.thermal_img())
        img_obs = np.expand_dims(img_obs, axis=0)

        return img_obs, reward, done, info


class IREnvWrapper(gym.Wrapper, Render):
    def __init__(self, env, args):
        super(IREnvWrapper, self).__init__(env)
        if args.render:
            Render.__init__(self, args)
            time.sleep(SIM_DELAY) # wating for simulator loading
        self.focal_length_90    = 110 # Not accurate
        self.focal_length_120   = 70 # Not accurate
        self.env                = env
        self.args               = args
        self.observation_space  = gym.spaces.Box(low=0, high=255,
                                            shape=(1, 86, 86), dtype=np.uint8)

    def reset(self):
        _, info = self.env.reset()
        if self.args.render:
            self.sim_reset()
        IR_90 = self.infrared_image(self.env.M_states, self.env.T_states, self.focal_length_90)
        IR_120 = self.infrared_image(self.env.M_states, self.env.T_states, self.focal_length_120)
        img_obs = np.expand_dims(IR_90, axis=0)
        
        return img_obs, info
    
    def render(self):
        pass

    def rotation_matrix(self, roll, pitch, yaw):
        R_roll = np.array([[1, 0, 0],
                            [0, np.cos(roll), np.sin(roll)],
                            [0, -np.sin(roll), np.cos(roll)]], dtype=np.float32)
        R_pitch = np.array([[np.cos(pitch), 0, -np.sin(pitch)],
                            [0, 1, 0],
                            [np.sin(pitch), 0, np.cos(pitch)]], dtype=np.float32)
        R_yaw = np.array([[np.cos(yaw), np.sin(yaw), 0],
                            [-np.sin(yaw), np.cos(yaw), 0],
                            [0, 0, 1]], dtype=np.float32)
        
        return R_roll, R_pitch, R_yaw
    
    def infrared_image(self, Mstates, Tstates, focal_length):
        '''
        2D kinematics
        Mstates: x, z, theta, a_z
        Tstates: x, z, theta, a_z 
        '''
        # missile states
        M_pos = np.zeros(3)
        M_pos[0] = Mstates[0]
        M_pos[2] = Mstates[1]

        M_roll = 0.0
        M_pitch = -Mstates[2]
        M_yaw = 0.0

        # target states
        T_pos = np.zeros(3)
        T_pos[0] = Tstates[0]
        T_pos[2] = Tstates[1]

        # Get rotation matrix
        M_R_roll, M_R_pitch, M_R_yaw  = self.rotation_matrix(M_roll, M_pitch, M_yaw)
        
        # Transformation to missile coordinate
        missile_target = np.dot(np.dot(np.dot(M_R_yaw, M_R_pitch), M_R_roll), T_pos - M_pos)
        
        # Transformation to seeker coordinate (Camera coordinate)
        Seeker_R_roll, Seeker_R_pitch, Seeker_R_yaw  = self.rotation_matrix(0, np.pi/2, -np.pi/2)
        seeker_target = np.dot(np.dot(np.dot(Seeker_R_yaw, Seeker_R_pitch), Seeker_R_roll), missile_target)
        
        # ========== FOV Camera ==========
        image_width = 320
        image_height = 240
        camera_matrix = np.array([[focal_length, 0, image_width / 2],
                                [0, focal_length, image_height / 2],
                                [0, 0, 1]], dtype=np.float32)
        
        # Project 3d point to camera plane
        point_2d, _ = cv.projectPoints(seeker_target, np.zeros(3), np.zeros(3), camera_matrix, np.zeros(4))

        # convert to int
        point_2d = tuple(map(int, point_2d.squeeze()))
        
        # background image
        image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        
        # Add target
        Range = LA.norm(M_pos - T_pos)
        radius = max(int((focal_length/10)*np.exp(-(Range**2)/(500**2))), 1)
        cv.circle(image, point_2d, radius, (255, 255, 255), -1)
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        image = cv.resize(image, (86,86), interpolation=cv.INTER_AREA)
        # cv.imshow('Focal length {}'.format(focal_length), image)
        # cv.waitKey(1) & 0xFF

        return image

    def step(self, action):
        time.sleep(STEP_DELAY) # Simulator time delay
        if self.args.render:
            self.operation(self.mode, self.sub_mode, self.str_method)
            self.missile(self.env.M_states, self.env.T_states, self.env.reward)
        obs, reward, done, info = self.env.step(action)
        IR_90 = self.infrared_image(self.env.M_states, self.env.T_states, self.focal_length_90)
        IR_120 = self.infrared_image(self.env.M_states, self.env.T_states, self.focal_length_120)
        IR_obs = np.expand_dims(IR_90, axis=0)

        return IR_obs, reward, done, info


# ========== [Image + State] Stacked observation environemnts ==========
class ImageStackEnvWrapper(gym.Wrapper, Render):
    def __init__(self, env, args):
        super(ImageStackEnvWrapper, self).__init__(env)
        if args.render:
            Render.__init__(self, args)
            time.sleep(SIM_DELAY) # wating for simulator loading
        self.env                = env
        self.args               = args

        # Observation
        self.obs_stack = 4 # sequential 4 frame stack
        spaces = {'vec': gym.spaces.Box(low=-1000, high = 1000,
                                        shape=(self.obs_stack*self.env.obs_dim,), dtype=np.float32),
                'img': gym.spaces.Box(low=0, high=255,
                                            shape=(2*self.obs_stack, 86, 86), dtype=np.uint8)}
        self.observation_space = gym.spaces.Dict(spaces)

    def reset(self, options = {}):
        state_obs, info = self.env.reset(options=options)
        self.sim_reset()

        # Get image
        IR_90 = self.img_processing(self.thermal_img())
        IR_120 = self.img_processing(self.thermal_img_2())

        self.state_queue = deque()
        self.image_queue = deque()

        for _ in range(self.obs_stack):
            self.state_queue.append(state_obs)
            self.image_queue.append(IR_90)
            self.image_queue.append(IR_120)

        obs = self.observation_space.sample()
        obs['vec'] = np.ravel(self.state_queue)
        obs['img'] = np.array(self.image_queue)

        return obs, info
    
    def img_processing(self, img):
        img = cv.resize(img, (86, 86), interpolation=cv.INTER_AREA)
        # img = jitter(crop(img))
        return img
        
    def render(self):
        pass

    def step(self, action):
        time.sleep(STEP_DELAY) # Simulator time delay
        self.operation(self.mode, self.sub_mode, self.str_method)
        self.missile(self.env.M_states, self.env.T_states, self.env.reward)
        state_obs, reward, done, info = self.env.step(action)
        IR_90 = self.img_processing(self.thermal_img())
        IR_120 = self.img_processing(self.thermal_img_2())

        # ========== Visualization ==========
        # cv.imshow('IR_90, Focal length {}'.format('70'), IR_90)
        # cv.imshow('IR_120, Focal length {}'.format('110'), IR_120)
        # cv.waitKey(1) & 0xFF

        # state queue
        self.state_queue.popleft()
        self.state_queue.append(state_obs)

        # image queue
        self.image_queue.popleft()
        self.image_queue.popleft()
        self.image_queue.append(IR_90)
        self.image_queue.append(IR_120)

        obs = self.observation_space.sample()
        obs['vec'] = np.ravel(self.state_queue)
        obs['img'] = np.array(self.image_queue)

        return obs, reward, done, info

    def close(self):
        self.sim.kill()


class IRStackEnvDynWrapper(gym.Wrapper, Render):
    def __init__(self, env, args):
        super(IRStackEnvDynWrapper, self).__init__(env)
        if args.render:
            Render.__init__(self, args)
            time.sleep(SIM_DELAY) # wating for simulator loading
        self.focal_length_90    = 110 # Not accurate
        self.focal_length_120   = 70 # Not accurate
        self.env                = env
        self.args               = args

        # Observation
        self.obs_stack = 4 # sequential 4 frame stack
        spaces = {'vec': gym.spaces.Box(low=-np.pi, high = np.pi,
                                        shape=(self.obs_stack*self.env.obs_dim,), dtype=np.float32),
                'img': gym.spaces.Box(low=0, high=255,
                                            shape=(2*self.obs_stack, 86, 86), dtype=np.uint8)}
        self.observation_space = gym.spaces.Dict(spaces)

    def reset(self, options={}):
        state_obs, info = self.env.reset(options=options)
        if self.args.render:
            self.sim_reset()
        IR_90 = self.infrared_image(self.env.M_states, self.env.T_states, self.focal_length_90)
        IR_120 = self.infrared_image(self.env.M_states, self.env.T_states, self.focal_length_120)
        
        self.state_queue = deque(maxlen=self.obs_stack)
        self.image_queue = deque(maxlen=self.obs_stack*2)

        for _ in range(self.obs_stack):
            self.state_queue.append(state_obs)
            self.image_queue.append(IR_90)
            self.image_queue.append(IR_120)
 
        obs = self.observation_space.sample()
        obs['vec'] = np.ravel(self.state_queue)
        obs['img'] = np.array(self.image_queue)
        
        return obs, info
    
    def render(self):
        pass

    def rotation_matrix(self, roll, pitch, yaw):
        R_roll = np.array([[1, 0, 0],
                            [0, np.cos(roll), np.sin(roll)],
                            [0, -np.sin(roll), np.cos(roll)]], dtype=np.float32)
        R_pitch = np.array([[np.cos(pitch), 0, -np.sin(pitch)],
                            [0, 1, 0],
                            [np.sin(pitch), 0, np.cos(pitch)]], dtype=np.float32)
        R_yaw = np.array([[np.cos(yaw), np.sin(yaw), 0],
                            [-np.sin(yaw), np.cos(yaw), 0],
                            [0, 0, 1]], dtype=np.float32)
        
        return R_roll, R_pitch, R_yaw
    
    def infrared_image(self, Mstates, Tstates, focal_length):
        '''
        2D kinematics
        Missile : X, Z, alpha, theta, q, delta, alpha_dot, theta_dot, q_dot, delta_dot
        Target  : X, Z, gamma, gamma_dot
        '''
        # missile states
        M_pos = np.zeros(3)
        M_pos[0] = Mstates[0]
        M_pos[2] = Mstates[1]

        M_roll = 0.0
        M_pitch = -Mstates[3]
        M_yaw = 0.0

        # target states
        T_pos = np.zeros(3)
        T_pos[0] = Tstates[0]
        T_pos[2] = Tstates[1]

        # Get rotation matrix
        M_R_roll, M_R_pitch, M_R_yaw  = self.rotation_matrix(M_roll, M_pitch, M_yaw)
        
        # Transformation to missile coordinate
        missile_target = np.dot(np.dot(np.dot(M_R_yaw, M_R_pitch), M_R_roll), T_pos - M_pos)
        
        # Transformation to seeker coordinate (Camera coordinate)
        Seeker_R_roll, Seeker_R_pitch, Seeker_R_yaw  = self.rotation_matrix(0, np.pi/2, -np.pi/2)
        seeker_target = np.dot(np.dot(np.dot(Seeker_R_yaw, Seeker_R_pitch), Seeker_R_roll), missile_target)
        
        # ========== FOV Camera ==========
        image_width = 320
        image_height = 240
        camera_matrix = np.array([[focal_length, 0, image_width / 2],
                                [0, focal_length, image_height / 2],
                                [0, 0, 1]], dtype=np.float32)
        
        # Project 3d point to camera plane
        point_2d, _ = cv.projectPoints(seeker_target, np.zeros(3), np.zeros(3), camera_matrix, np.zeros(4))

        # convert to int
        point_2d = tuple(map(int, point_2d.squeeze()))
        
        # background image
        image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        
        # Add target
        Range = LA.norm(M_pos - T_pos)
        radius = max(int((2*focal_length/10)*np.exp(-(Range**2)/(1000**2))), 1)
        cv.circle(image, point_2d, radius, (255, 255, 255), -1)
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        image = cv.resize(image, (86,86), interpolation=cv.INTER_AREA)
        # print(point_2d)

        return image

    def step(self, action):
        if self.args.render:
            time.sleep(STEP_DELAY) # Simulator time delay
            self.operation(self.mode, self.sub_mode, self.str_method)
            self.missile(self.env.M_states, self.env.T_states, self.env.reward)
        state_obs, reward, done, info = self.env.step(action)
        IR_90 = self.infrared_image(self.env.M_states, self.env.T_states, self.focal_length_90)
        IR_120 = self.infrared_image(self.env.M_states, self.env.T_states, self.focal_length_120)

        # ========== Visualization ==========
        # cv.imshow('IR_90, Focal length {}'.format(self.focal_length_90), IR_90)
        # cv.imshow('IR_120, Focal length {}'.format(self.focal_length_120), IR_120)
        # cv.waitKey(1) & 0xFF

        # state queue
        self.state_queue.popleft()
        self.state_queue.append(state_obs)

        # image queue
        self.image_queue.popleft()
        self.image_queue.popleft()
        self.image_queue.append(IR_90)
        self.image_queue.append(IR_120)

        obs = self.observation_space.sample()
        obs['vec'] = np.ravel(self.state_queue)
        obs['img'] = np.array(self.image_queue)

        return obs, reward, done, info


class IRStackEnvWrapper(gym.Wrapper, Render):
    def __init__(self, env, args):
        super(IRStackEnvWrapper, self).__init__(env)
        if args.render:
            Render.__init__(self, args)
            time.sleep(SIM_DELAY) # wating for simulator loading
        self.focal_length_90    = 110 # Not accurate
        self.focal_length_120   = 70 # Not accurate
        self.env                = env
        self.args               = args

        # Observation
        self.obs_stack = 4 # sequential 4 frame stack
        spaces = {'vec': gym.spaces.Box(low=-1000, high = 1000,
                                        shape=(self.obs_stack*self.env.obs_dim,), dtype=np.float32),
                'img': gym.spaces.Box(low=0, high=255,
                                            shape=(2*self.obs_stack, 86, 86), dtype=np.uint8)}
        self.observation_space = gym.spaces.Dict(spaces)
        self.env_dim = 2 if self.env.obs_dim == 5 else 3

    def reset(self, options: dict = {}):
        state_obs, info = self.env.reset(options=options)
        if self.args.render:
            self.sim_reset()
        IR_90 = self.infrared_image(self.env.M_states, self.env.T_states, self.focal_length_90)
        IR_120 = self.infrared_image(self.env.M_states, self.env.T_states, self.focal_length_120)
        
        self.state_queue = deque()
        self.image_queue = deque()

        for _ in range(self.obs_stack):
            self.state_queue.append(state_obs)
            self.image_queue.append(IR_90)
            self.image_queue.append(IR_120)
 
        obs = self.observation_space.sample()
        obs['vec'] = np.ravel(self.state_queue)
        obs['img'] = np.array(self.image_queue)
        
        return obs, info
    
    def render(self):
        pass

    def rotation_matrix(self, roll, pitch, yaw):
        R_roll = np.array([[1, 0, 0],
                            [0, np.cos(roll), np.sin(roll)],
                            [0, -np.sin(roll), np.cos(roll)]], dtype=np.float32)
        R_pitch = np.array([[np.cos(pitch), 0, -np.sin(pitch)],
                            [0, 1, 0],
                            [np.sin(pitch), 0, np.cos(pitch)]], dtype=np.float32)
        R_yaw = np.array([[np.cos(yaw), np.sin(yaw), 0],
                            [-np.sin(yaw), np.cos(yaw), 0],
                            [0, 0, 1]], dtype=np.float32)
        
        return R_roll, R_pitch, R_yaw
    
    def infrared_image(self, Mstates, Tstates, focal_length):
        '''
        2D kinematics
        Mstates: x, z, theta, a_z
        Tstates: x, z, theta, a_z
        '''

        if self.env_dim == 2:
            # missile states
            M_pos = np.zeros(3)
            M_pos[0] = Mstates[0]
            M_pos[2] = Mstates[1]

            M_roll = 0.0
            M_pitch = -Mstates[2]
            M_yaw = 0.0

            # target states
            T_pos = np.zeros(3)
            T_pos[0] = Tstates[0]
            T_pos[2] = Tstates[1]
        
        elif self.env_dim == 3:
            M_pos = Mstates[:3]
            M_roll = Mstates[6]
            M_pitch = Mstates[7]
            M_yaw = Mstates[8]
            T_pos = Tstates[:3]

        # Get rotation matrix
        M_R_roll, M_R_pitch, M_R_yaw  = self.rotation_matrix(M_roll, M_pitch, M_yaw)
        
        # Transformation to missile coordinate
        missile_target = np.dot(np.dot(np.dot(M_R_yaw, M_R_pitch), M_R_roll), T_pos - M_pos)
        
        # Transformation to seeker coordinate (Camera coordinate)
        Seeker_R_roll, Seeker_R_pitch, Seeker_R_yaw  = self.rotation_matrix(0, np.pi/2, -np.pi/2)
        seeker_target = np.dot(np.dot(np.dot(Seeker_R_yaw, Seeker_R_pitch), Seeker_R_roll), missile_target)
        
        # ========== FOV Camera ==========
        image_width = 320
        image_height = 240
        camera_matrix = np.array([[focal_length, 0, image_width / 2],
                                [0, focal_length, image_height / 2],
                                [0, 0, 1]], dtype=np.float32)
        
        # Project 3d point to camera plane
        point_2d, _ = cv.projectPoints(seeker_target, np.zeros(3), np.zeros(3), camera_matrix, np.zeros(4))

        # convert to int
        point_2d = tuple(map(int, point_2d.squeeze()))
        
        # background image
        image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        
        # Add target
        Range = LA.norm(M_pos - T_pos)
        radius = max(int((2*focal_length/10)*np.exp(-(Range**2)/(500**2))), 1)
        cv.circle(image, point_2d, radius, (255, 255, 255), -1)
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        image = cv.resize(image, (86,86), interpolation=cv.INTER_AREA)
        # image = jitter(crop(crop(image)))

        return image

    def step(self, action):
        if self.args.render:
            time.sleep(STEP_DELAY) # Simulator time delay
            self.operation(self.mode, self.sub_mode, self.str_method)
            self.missile(self.env.M_states, self.env.T_states, self.env.reward)
        state_obs, reward, done, info = self.env.step(action)
        IR_90 = self.infrared_image(self.env.M_states, self.env.T_states, self.focal_length_90)
        IR_120 = self.infrared_image(self.env.M_states, self.env.T_states, self.focal_length_120)

        # ========== Visualization ==========
        # cv.imshow('IR_90, Focal length {}'.format(self.focal_length_90), IR_90)
        # cv.imshow('IR_120, Focal length {}'.format(self.focal_length_120), IR_120)
        # cv.waitKey(1) & 0xFF

        # state queue
        self.state_queue.popleft()
        self.state_queue.append(state_obs)

        # image queue
        self.image_queue.popleft()
        self.image_queue.popleft()
        self.image_queue.append(IR_90)
        self.image_queue.append(IR_120)

        obs = self.observation_space.sample()
        obs['vec'] = np.ravel(self.state_queue)
        obs['img'] = np.array(self.image_queue)

        # check if target is in camera frames
        # if self.env_dim == 3:
        #     lost = info['LOS_t']
        #     losg = info['LOS_g']
        #     # lost = info['guidance'][2]
        #     # losg = info['guidance'][3]
        #     if (abs(lost) >= np.pi / 3) or (abs(losg) >= np.pi / 3):
        #         done = True

        return obs, reward, done, info
    
    def close(self):
        self.sim.kill()

class VectorWrapper(gym.ObservationWrapper):

    def __init__(
        self,
        env: gym.Env,
        args: Namespace,
    ):
        super().__init__(env)

        self.num_stack = 4
        self.frames = deque(maxlen=self.num_stack)

        low = np.repeat(self.observation_space.low, self.num_stack)
        high = np.repeat(self.observation_space.high, self.num_stack)

        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def observation(self, observation):
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        return np.concatenate(list(self.frames), axis=-1)

    def step(self, action):
        obs, rwd, done, info = self.env.step(action)
        self.frames.append(obs)
        return self.observation(None), rwd, done, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        [self.frames.append(obs) for _ in range(self.num_stack)]
        return self.observation(None), info


class ImageWrapper(gym.ObservationWrapper):

    def __init__(
            self,
            env: gym.Env,
    ):
        super().__init__(env)
        self.observation_space = self.observation_space['img']

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs['img'], info

    def step(self, action):
        obs, rwd, done, info = self.env.step(action)
        return obs['img'], rwd, done, info

