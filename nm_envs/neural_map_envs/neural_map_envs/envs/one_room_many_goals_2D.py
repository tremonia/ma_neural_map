import gym
from gym import error, spaces, utils

import numpy as np


class One_Room_Many_Goals_2D(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, map_size_vertical, map_size_horizontal, no_goals, view_straightforward, view_left_right, start_position_x, start_position_y, start_orientation, reward_range, max_no_steps):

        # set some parameters
        self.map_size_vertical = map_size_vertical
        self.map_size_horizontal = map_size_horizontal
        self.no_goals = no_goals
        self.view_straightforward = view_straightforward
        self.view_left_right = view_left_right
        self.reward_range = reward_range
        self.max_no_steps = max_no_steps
        self.living_reward = 0.5 / self.max_no_steps

        # determine goal rewards
        if self.no_goals == 2:
            self.goal_rewards = [0.25, 0.75]
        elif self.no_goals == 3:
            self.goal_rewards = [0.125, 0.125, 0.75]
        elif self.no_goals == 4:
            self.goal_rewards = [0.1, 0.1, 0.1, 0.7]

        # build map with dummy surrounding
        self.map = np.zeros((self.no_goals + 1, map_size_vertical + 2*(self.view_straightforward-1), map_size_horizontal +2*(self.view_straightforward-1)), dtype=np.float32)
        self.map[0,:,:] = 1
        self.map[0, (self.view_straightforward-1):(1-self.view_straightforward), (self.view_straightforward-1):(1-self.view_straightforward)] = 0

        # shift positions due to dummy surrounding
        self.start_position_x = self.position_x = start_position_x + self.view_straightforward -1
        self.start_position_y = self.position_y = start_position_y + self.view_straightforward -1
        self.start_orientation = self.orientation = start_orientation  # down=0, right=1, up=2, left=3

        # determine random goal positions and place corresponding indicators in the map
        self.start_position_index = (start_position_y * map_size_horizontal) + start_position_x
        self.max_goal_index = (map_size_horizontal * map_size_vertical) - 1

        goal1_index = np.random.random_integers(0, self.max_goal_index)
        while goal1_index == self.start_position_index:
            goal1_index = np.random.random_integers(0, self.max_goal_index)
        self.goal1_position_x = (goal1_index % self.map_size_horizontal) + self.view_straightforward -1
        self.goal1_position_y = (goal1_index // self.map_size_vertical) + self.view_straightforward -1
        self.map[1, self.goal1_position_y, self.goal1_position_x] = 1.

        goal2_index = np.random.random_integers(0, self.max_goal_index)
        while goal2_index == goal1_index:
            goal2_index = np.random.random_integers(0, self.max_goal_index)
        self.goal2_position_x = (goal2_index % self.map_size_horizontal) + self.view_straightforward -1
        self.goal2_position_y = (goal2_index // self.map_size_vertical) + self.view_straightforward -1
        self.map[2, self.goal2_position_y, self.goal2_position_x] = 1.

        goal3_index = np.random.random_integers(0, self.max_goal_index)
        while goal3_index == goal2_index:
            goal3_index = np.random.random_integers(0, self.max_goal_index)
        self.goal3_position_x = (goal3_index % self.map_size_horizontal) + self.view_straightforward -1
        self.goal3_position_y = (goal3_index // self.map_size_vertical) + self.view_straightforward -1
        if self.no_goals > 2:
            self.map[3, self.goal3_position_y, self.goal3_position_x] = 1.

        goal4_index = np.random.random_integers(0, self.max_goal_index)
        while goal4_index == goal3_index:
            goal4_index = np.random.random_integers(0, self.max_goal_index)
        self.goal4_position_x = (goal4_index % self.map_size_horizontal) + self.view_straightforward -1
        self.goal4_position_y = (goal4_index // self.map_size_vertical) + self.view_straightforward -1
        if self.no_goals > 3:
            self.map[4, self.goal4_position_y, self.goal4_position_x] = 1.

        # set observation and action space
        observation_shape = (self.map.shape[0], self.view_straightforward, 2*self.view_left_right+1)
        self.observation_space = spaces.Dict({"position": spaces.MultiDiscrete([self.map.shape[2], self.map.shape[1], 4]),
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
            return {"position" : np.array((int(self.position_x), int(self.position_y), int(self.orientation))), "observation" : self.last_obs}, 0., self.last_done, {}
        else:
            # if step_counter reaches its maximum, terminate episode
            if self.step_counter == self.max_no_steps:
                self.last_done = True
                return {"position" : np.array((int(self.position_x), int(self.position_y), int(self.orientation))), "observation" : self.last_obs}, -0.5, self.last_done, {}
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
                if (self.position_x == self.goal1_position_x) and (self.position_y == self.goal1_position_y) and (self.internal_state == 'Empty'):
                    reward = self.goal_rewards[0]
                    self.last_done = False
                    self.internal_state = 'FoundGoal1'

                elif (self.position_x == self.goal2_position_x) and (self.position_y == self.goal2_position_y) and (self.internal_state == 'FoundGoal1'):
                    reward = self.goal_rewards[1]
                    if self.no_goals == 2:
                        self.last_done = True
                    else:
                        self.last_done = False
                    self.internal_state = 'FoundGoal2'

                elif (self.position_x == self.goal3_position_x) and (self.position_y == self.goal3_position_y) and (self.internal_state == 'FoundGoal2'):
                    reward = self.goal_rewards[2]
                    if self.no_goals == 3:
                        self.last_done = True
                    else:
                        self.last_done = False
                    self.internal_state = 'FoundGoal3'

                elif (self.position_x == self.goal4_position_x) and (self.position_y == self.goal4_position_y) and (self.internal_state == 'FoundGoal3'):
                    reward = self.goal_rewards[3]
                    self.last_done = True
                    self.internal_state = 'FoundGoal4'

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
            reward -= self.living_reward
            #if self.step_counter == 1.:
            #    return {"position": np.array((int(self.position_x), int(self.position_y), int(self.orientation))), "observation": self.last_obs}, reward, self.last_done, {'goal_positions': [self.goal1_position_x, self.goal1_position_y, self.goal2_position_x, self.goal2_position_y, self.goal3_position_x, self.goal3_position_y, self.goal4_position_x, self.goal4_position_y]}
            #else:
            return {"position": np.array((int(self.position_x), int(self.position_y), int(self.orientation))), "observation": self.last_obs}, reward, self.last_done, {}

    def reset(self):
        # first delete / reset old goal indicators
        self.map[1, self.goal1_position_y, self.goal1_position_x] = 0.
        self.map[2, self.goal2_position_y, self.goal2_position_x] = 0.
        if self.no_goals > 2:
            self.map[3, self.goal3_position_y, self.goal3_position_x] = 0.
        if self.no_goals > 3:
            self.map[4, self.goal4_position_y, self.goal4_position_x] = 0.

        # determine random goal positions and place corresponding indicators in the map
        goal1_index = np.random.random_integers(0, self.max_goal_index)
        while goal1_index==self.start_position_index:
            goal1_index = np.random.random_integers(0, self.max_goal_index)
        self.goal1_position_x = (goal1_index % self.map_size_horizontal) + self.view_straightforward -1
        self.goal1_position_y = (goal1_index // self.map_size_vertical) + self.view_straightforward -1
        self.map[1, self.goal1_position_y, self.goal1_position_x] = 1.

        goal2_index = np.random.random_integers(0, self.max_goal_index)
        while goal2_index==goal1_index:
            goal2_index = np.random.random_integers(0, self.max_goal_index)
        self.goal2_position_x = (goal2_index % self.map_size_horizontal) + self.view_straightforward -1
        self.goal2_position_y = (goal2_index // self.map_size_vertical) + self.view_straightforward -1
        self.map[2, self.goal2_position_y, self.goal2_position_x] = 1.

        goal3_index = np.random.random_integers(0, self.max_goal_index)
        while goal3_index==goal2_index:
            goal3_index = np.random.random_integers(0, self.max_goal_index)
        self.goal3_position_x = (goal3_index % self.map_size_horizontal) + self.view_straightforward -1
        self.goal3_position_y = (goal3_index // self.map_size_vertical) + self.view_straightforward -1
        if self.no_goals > 2:
            self.map[3, self.goal3_position_y, self.goal3_position_x] = 1.

        goal4_index = np.random.random_integers(0, self.max_goal_index)
        while goal4_index == goal3_index:
            goal4_index = np.random.random_integers(0, self.max_goal_index)
        self.goal4_position_x = (goal4_index % self.map_size_horizontal) + self.view_straightforward -1
        self.goal4_position_y = (goal4_index // self.map_size_vertical) + self.view_straightforward -1
        if self.no_goals > 3:
            self.map[4, self.goal4_position_y, self.goal4_position_x] = 1.

        # reset internal variables
        self.position_x = self.start_position_x
        self.position_y = self.start_position_y
        self.orientation = self.start_orientation

        self.last_done = False

        self.internal_state = 'Empty'
        self.step_counter = 0.

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

        return {"position": np.array((int(self.position_x), int(self.position_y), int(self.orientation))), "observation": self.last_obs}


    def render(self, mode='human'):
        print('rendering has to be implemented...')
