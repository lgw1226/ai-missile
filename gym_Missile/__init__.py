from gym.envs.registration import register

# ==================== State Based Env ====================
register(
    id='kinematics2d-v2',
    entry_point='gym_Missile.envs:Kinematics2d_Env_v2'
)

register(
    id='kinematics2d-v1',
    entry_point='gym_Missile.envs:Kinematics2d_Env_v1'
)

register(
    id='kinematics2d-v0',
    entry_point='gym_Missile.envs:Kinematics2d_Env'
)

register(
    id='kinematics3d-v0',
    entry_point='gym_Missile.envs:Kinematics3d_Env',
)

register(
    id='kinematics3d-v1',
    entry_point='gym_Missile.envs:Kinematics3d_Env_v1',
)

register(
    id='dynamics2d-v0',
    entry_point='gym_Missile.envs:Dynamics2d_Env'
)