import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np


class One_Key_Three_Boxes_2D(gym.Env):
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

        # determine key and goal positions
        self.key_position_x = np.nonzero(self.map[1,:,:])[1][0]
        self.key_position_y = np.nonzero(self.map[1,:,:])[0][0]

        self.box1_position_x = np.nonzero(self.map[2,:,:])[1][0]
        self.box1_position_y = np.nonzero(self.map[2,:,:])[0][0]

        self.box2_position_x = np.nonzero(self.map[3,:,:])[1][0]
        self.box2_position_y = np.nonzero(self.map[3,:,:])[0][0]

        self.box3_position_x = np.nonzero(self.map[4,:,:])[1][0]
        self.box3_position_y = np.nonzero(self.map[4,:,:])[0][0]

        # set observation and action space
        observation_shape = (self.map.shape[0], self.view_straightforward, 2*self.view_left_right+1)
        self.observation_space = spaces.Dict({"position": spaces.MultiDiscrete([self.map.shape[2], self.map.shape[1]]),
                                              "observation": spaces.Box(low=0., high=1., shape=observation_shape, dtype=np.float32)})
        self.action_space = spaces.Discrete(3)

        self.last_done = False
        self.last_obs = np.zeros(observation_shape)

        # init internal state and step counter
        self.internal_state = 'Empty'
        self.step_counter = 0.

        #self.last_action = None  # for rendering





    def step(self, action):
        # don't do anything if previous state was a terminal state
        if self.last_done:
            return {"position" : np.array((int(self.position_x), int(self.position_y))), "observation" : self.last_obs}, 0., self.last_done, {}
        else:
            # if step_counter reaches its maximum, terminate episode
            if self.step_counter == 1000.:
                self.last_done = True
                reward = -1.
                #print('\nStep counter reached maximum!')
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

                # determine reward, done and internal_state
                if self.position_x == self.key_position_x and self.position_y == self.key_position_y:
                    if self.internal_state == 'Empty':
                        #print('\nFound Key in correct way')
                        reward = 0.5
                        self.last_done = False
                        self.internal_state = 'FoundKey'
                    else:
                        reward = 0.
                        self.last_done = False

                elif self.position_x == self.box1_position_x and self.position_y == self.box1_position_y:
                    if self.internal_state == 'FoundKey':
                        #print('\nFound Box1 in correct way')
                        reward = 0.5
                        self.last_done = False
                        self.internal_state = 'FoundBox1'
                    else:
                        reward = 0.
                        self.last_done = False

                elif self.position_x == self.box2_position_x and self.position_y == self.box2_position_y:
                    if self.internal_state == 'FoundBox1':
                        #print('\nFound Box2 in correct way')
                        reward = 0.5
                        self.last_done = False
                        self.internal_state = 'FoundBox2'
                    else:
                        reward = 0.0
                        self.last_done = False

                elif self.position_x == self.box3_position_x and self.position_y == self.box3_position_y:
                    if self.internal_state == 'FoundBox2':
                        #print('\nFound Box3 in correct way')
                        reward = 1.
                        self.last_done = True
                        self.internal_state = 'FoundBox3'
                    else:
                        reward = 0.
                        self.last_done = False

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

            self.step_counter += 1.
            reward -= 0.001

            return {"position": np.array((int(self.position_x), int(self.position_y))), "observation": self.last_obs}, reward, self.last_done, {}



    def reset(self):
        self.position_x = self.start_position_x
        self.position_y = self.start_position_y
        self.orientation = self.start_orientation

        self.last_done = False

        self.internal_state = 'Empty'
        self.step_counter = 0.

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
