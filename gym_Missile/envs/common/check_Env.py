import gym
import gym_Missile
from stable_baselines3.common.env_checker import check_env

def main():
    
    env = gym.make('kinematics2d-v1')
    check_env(env)

    done_count = 0
    obs = env.reset()
    while done_count != 1:

        obs, reward, done, info = env.step(0)
        if done:
            obs = env.reset()
            done_count += 1
            print("Done!!")
    env.close()

if __name__ == "__main__":
    main()
