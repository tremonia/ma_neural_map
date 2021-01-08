import gym
import numpy as np

from gym import spaces

map_tmp = np.load('EasyMap001.npy')

env = gym.make('neural_map_envs:two_indicators_goals_dimensions-v0')

obs = env.reset()

obs, reward, done, _ = env.step(1)



print(map_tmp[0,:,:])
print(obs['observation'][0,:,:])
print('Position: ', obs['position'])
print('Reward: ', reward)
print('Done: ', done)
