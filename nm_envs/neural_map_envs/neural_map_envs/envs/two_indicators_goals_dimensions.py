import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np


class Two_Goals_Indicators_Dimensions(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, map_name, view_straightforward, view_left_right, start_position_x, start_position_y, start_orientation, reward_range):
        # load map and set some parameters
        map_tmp = np.load(map_name)
        self.view_straightforward = view_straightforward
        self.view_left_right = view_left_right
        self.reward_range = reward_range

        # build dummy surrounding
        self.map = np.zeros((map_tmp.shape[0], map_tmp.shape[1] + 2*(self.view_straightforward-1), map_tmp.shape[2] +2*(self.view_straightforward-1)), dtype=np.float32)
        self.map[0,:,:] = 1
        self.map[:, (self.view_straightforward-1):(1-self.view_straightforward), (self.view_straightforward-1):(1-self.view_straightforward)] = map_tmp

        # shift positions due to dummy surrounding
        self.start_position_x = self.position_x = start_position_x + self.view_straightforward -1
        self.start_position_y = self.position_y = start_position_y + self.view_straightforward -1
        self.start_orientation = self.orientation = start_orientation  # down=0, right=1, up=2, left=3

        # determine wright and wrong goal position
        if np.any(self.map[1,:,:]):
            #Indicator 1 -> wright goal position in layer 3, wrong in layer 4
            self.indicator_position_x = np.nonzero(self.map[1,:,:])[1][0]
            self.indicator_position_y = np.nonzero(self.map[1,:,:])[0][0]
            self.wright_goal_position_x = np.nonzero(self.map[3,:,:])[1][0]
            self.wright_goal_position_y = np.nonzero(self.map[3,:,:])[0][0]
            self.wrong_goal_position_x = np.nonzero(self.map[4,:,:])[1][0]
            self.wrong_goal_position_y = np.nonzero(self.map[4,:,:])[0][0]
        elif np.any(self.map[2,:,:]):
            # Indicator 2 -> wright goal position in layer 4, wrong in layer 3
            self.indicator_position_x = np.nonzero(self.map[2,:,:])[1][0]
            self.indicator_position_y = np.nonzero(self.map[2,:,:])[0][0]
            self.wright_goal_position_x = np.nonzero(self.map[4,:,:])[1][0]
            self.wright_goal_position_y = np.nonzero(self.map[4,:,:])[0][0]
            self.wrong_goal_position_x = np.nonzero(self.map[3,:,:])[1][0]
            self.wrong_goal_position_y = np.nonzero(self.map[3,:,:])[0][0]

        # set observation and action space
        observation_shape = (self.map.shape[0], self.view_straightforward, 2*self.view_left_right+1)
        self.observation_space = spaces.Dict({"position": spaces.MultiDiscrete([self.map.shape[2], self.map.shape[1]]),
                                              "observation": spaces.Box(low=0., high=1., shape=observation_shape, dtype=np.float32)})
        self.action_space = spaces.Discrete(3)

        self.last_done = False
        self.last_obs = np.zeros(observation_shape)

        #self.last_action = None  # for rendering



    def step(self, action):
        # don't do anything if previous state was a terminal state
        if self.last_done:
            return {"position" : np.array((int(self.position_x), int(self.position_y))), "observation" : self.last_obs}, 0., self.last_done, {}
        else:
            # perform action, i.e. adjust self.position_x, self.position_y, self.orientation
            if action == 0 or action == 2:
                self.orientation = (self.orientation + action - 1) % 4
            elif action == 1:
                if self.orientation == 0 and self.map[0, self.position_y + 1, self.position_x] == 0.:
                    self.position_y = self.position_y + 1
                elif self.orientation == 1 and self.map[0, self.position_y, self.position_x + 1] == 0.:
                    self.position_x = self.position_x + 1
                elif self.orientation == 2 and self.map[0, self.position_y - 1, self.position_x] == 0.:
                    self.position_y = self.position_y - 1
                elif self.orientation == 3 and self.map[0, self.position_y, self.position_x - 1] == 0.:
                    self.position_x = self.position_x - 1
            else:
                raise ValueError

            # determine reward and done
            if self.position_x == self.wright_goal_position_x and self.position_y == self.wright_goal_position_y:
                reward = 1.
                self.last_done = True
            elif self.position_x == self.wrong_goal_position_x and self.position_y == self.wrong_goal_position_y:
                reward = -1.
                self.last_done = True
            else:
                reward = 0.
                self.last_done = False

            # determine obs
            if self.orientation == 0:
                y_low = self.position_y
                y_high = self.position_y + self.view_straightforward
                x_low = self.position_x - self.view_left_right
                x_high = self.position_x + self.view_left_right + 1
                self.last_obs = np.flip(self.map[:, y_low:y_high, x_low:x_high], 2)
            elif self.orientation == 1:
                y_low = self.position_y - self.view_left_right
                y_high = self.position_y + self.view_left_right + 1
                x_low = self.position_x
                x_high = self.position_x + self.view_straightforward
                self.last_obs = np.transpose(self.map[:, y_low:y_high, x_low:x_high], [0, 2, 1])
            elif self.orientation == 2:
                y_low = self.position_y - self.view_straightforward + 1
                y_high = self.position_y + 1
                x_low = self.position_x - self.view_left_right
                x_high = self.position_x + self.view_left_right + 1
                self.last_obs = np.flip(self.map[:, y_low:y_high, x_low:x_high], 1)
            elif self.orientation == 3:
                y_low = self.position_y - self.view_left_right
                y_high = self.position_y + self.view_left_right + 1
                x_low = self.position_x - self.view_straightforward + 1
                x_high = self.position_x + 1
                self.last_obs = np.flip(np.transpose(np.flip(self.map[:, y_low:y_high, x_low:x_high], 2), [0, 2, 1]), 2)

            return {"position": np.array((int(self.position_x), int(self.position_y))), "observation": self.last_obs}, reward, self.last_done, {}



    def reset(self):
        # determine next indicator (layer 1 or 2) randomly
        if np.random.random_sample() < 0.5:
            # next indicator in layer 1
            self.map[1, self.indicator_position_y, self.indicator_position_x] = 1.
            self.map[2, self.indicator_position_y, self.indicator_position_x] = 0.
        else:
            # next indicator in layer 2
            self.map[1, self.indicator_position_y, self.indicator_position_x] = 0.
            self.map[2, self.indicator_position_y, self.indicator_position_x] = 1.

        # determine wright and wrong goal position
        if np.any(self.map[1,:,:]):
            #Indicator 1 -> wright goal position in layer 3, wrong in layer 4
            self.wright_goal_position_x = np.nonzero(self.map[3,:,:])[1][0]
            self.wright_goal_position_y = np.nonzero(self.map[3,:,:])[0][0]
            self.wrong_goal_position_x = np.nonzero(self.map[4,:,:])[1][0]
            self.wrong_goal_position_y = np.nonzero(self.map[4,:,:])[0][0]
        elif np.any(self.map[2,:,:]):
            # Indicator 2 -> wright goal position in layer 4, wrong in layer 3
            self.wright_goal_position_x = np.nonzero(self.map[4,:,:])[1][0]
            self.wright_goal_position_y = np.nonzero(self.map[4,:,:])[0][0]
            self.wrong_goal_position_x = np.nonzero(self.map[3,:,:])[1][0]
            self.wrong_goal_position_y = np.nonzero(self.map[3,:,:])[0][0]


        self.position_x = self.start_position_x
        self.position_y = self.start_position_y
        self.orientation = self.start_orientation

        self.last_done = False

        if self.orientation == 0:
            y_low = self.position_y
            y_high = self.position_y + self.view_straightforward
            x_low = self.position_x - self.view_left_right
            x_high = self.position_x + self.view_left_right + 1
            self.last_obs = np.flip(self.map[:, y_low:y_high, x_low:x_high], 2)
        elif self.orientation == 1:
            y_low = self.position_y - self.view_left_right
            y_high = self.position_y + self.view_left_right + 1
            x_low = self.position_x
            x_high = self.position_x + self.view_straightforward
            self.last_obs = np.transpose(self.map[:, y_low:y_high, x_low:x_high], [0, 2, 1])
        elif self.orientation == 2:
            y_low = self.position_y - self.view_straightforward + 1
            y_high = self.position_y + 1
            x_low = self.position_x - self.view_left_right
            x_high = self.position_x + self.view_left_right + 1
            self.last_obs = np.flip(self.map[:, y_low:y_high, x_low:x_high], 1)
        elif self.orientation == 3:
            y_low = self.position_y - self.view_left_right
            y_high = self.position_y + self.view_left_right + 1
            x_low = self.position_x - self.view_straightforward + 1
            x_high = self.position_x + 1
            self.last_obs = np.flip(np.transpose(np.flip(self.map[:, y_low:y_high, x_low:x_high], 2), [0, 2, 1]), 2)

        return {"position": np.array((int(self.position_x), int(self.position_y))), "observation": self.last_obs}

    def render(self, mode='human'):
        print('rendering has to be implemented...')
