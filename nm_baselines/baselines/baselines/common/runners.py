import numpy as np
from abc import ABC, abstractmethod

class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, nsteps, use_nm_customization=False, max_positions=[10, 10]):
        self.env = env
        self.model = model
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.nsteps = nsteps
        self.dones = [False for _ in range(nenv)]
        self.use_nm_customization = use_nm_customization
        self.states = model.initial_state

        if use_nm_customization:
            print('AbstractEnvRunner class uses neural map’s customization')
            self.batch_ob_shape = (nenv*nsteps,) + (env.observation_space.shape[0]-2,)
            self.obs = np.zeros((nenv,) + (env.observation_space.shape[0]-2,), dtype=env.observation_space.dtype.name)
            self.pos = np.zeros((nenv,) + (2,), dtype=env.observation_space.dtype.name)
            tmp = env.reset()
            self.obs = tmp[:,:-2]
            self.pos = tmp[:,-2:]

            # Init neural map's internal memory
            if model.initial_state is not None:
                print('AbstractEnvRunner class creates neural map’s internal memory')
                self.neural_map = model.initial_state
                self.neural_map_xy = np.zeros((self.neural_map.shape[0], self.neural_map.shape[3]), dtype=self.neural_map.dtype.name)
                self.pos_x_divisor = int(np.ceil(max_positions[0]/self.neural_map.shape[1]))
                self.pos_y_divisor = int(np.ceil(max_positions[1]/self.neural_map.shape[2]))
        else:
            self.batch_ob_shape = (nenv*nsteps,) + env.observation_space.shape
            self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
            self.obs[:] = env.reset()


    @abstractmethod
    def run(self):
        raise NotImplementedError
