import numpy as np
from abc import ABC, abstractmethod

class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, nsteps):
        self.env = env
        self.model = model
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        #CHANGES_START
        self.batch_ob_shape = (nenv*nsteps,) + (env.observation_space.shape[0]-2,)
        self.obs = np.zeros((nenv,) + (env.observation_space.shape[0],), dtype=env.observation_space.dtype.name)
        self.pos = np.zeros((nenv,) + (2,), dtype=env.observation_space.dtype.name)
        tmp = env.reset()       
        self.obs = tmp[:,:-2]
        self.pos = tmp[:,-2:]
        self.nsteps = nsteps
        self.states = model.initial_state

        # Init neural map's internal memory
        self.neural_map = model.initial_state
        self.neural_map_xy = np.zeros((self.neural_map.shape[0], self.neural_map.shape[3]), dtype=self.neural_map.dtype.name)
        #CHANGES_END
        self.dones = [False for _ in range(nenv)]

    @abstractmethod
    def run(self):
        raise NotImplementedError
