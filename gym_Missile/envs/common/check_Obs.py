import gym
import gym_Missile

env = gym.make('dynamics2d-v0')

state = env.reset()

print(f'Checking if the state is part of the observation space: {env.observation_space.contains(state)}')

# Checking if subsequent states are too.
done = False
while not done:
    state, _, done, _ = env.step(env.action_space.sample())
    print(f'Checking if the state is part of the observation space: {env.observation_space.contains(state)}')

env.close()