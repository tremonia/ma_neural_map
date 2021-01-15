import numpy as np
from baselines.common.runners import AbstractEnvRunner

class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam, use_nm_customization=False):
        super().__init__(env=env, model=model, nsteps=nsteps, use_nm_customization=use_nm_customization)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        if self.use_nm_customization:
            mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, mb_pos, mb_nm, mb_nm_xy = [],[],[],[],[],[],[],[],[]
        else:
            mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
            mb_states = self.states
        epinfos = []
        # For n in range number of steps
        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init

            if self.use_nm_customization:
                # Prepare nm_xy
                if self.model.initial_state is not None:
                    for i in range(self.neural_map.shape[0]):
                        self.neural_map_xy[i,:] = self.neural_map[i, int(self.pos[i,1]//2), int(self.pos[i,0]//2), :]

                actions, values, write_vector, neglogpacs = self.model.step(self.obs, S=self.neural_map, M=self.neural_map_xy)
            else:
                actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)

            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            if self.use_nm_customization:
                mb_pos.append(self.pos.copy())
                if self.model.initial_state is not None:
                    mb_nm.append(self.neural_map.copy())
                    mb_nm_xy.append(self.neural_map_xy.copy())

                    # Update neural map with write vector
                    for i in range(self.neural_map.shape[0]):
                        self.neural_map[i, int(self.pos[i,1]//2), int(self.pos[i,0]//2), :] = write_vector[i,:]

                # Take actions in env and look the results
                # Infos contains a ton of useful informations

                tmp, rewards, self.dones, infos = self.env.step(actions)
                self.obs = tmp[:,:-2]
                self.pos = tmp[:,-2:]
            else:
                self.obs[:], rewards, self.dones, infos = self.env.step(actions)

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)

        if self.use_nm_customization:
            mb_pos = np.asarray(mb_pos, dtype=self.pos.dtype)
            if self.model.initial_state is not None:
                mb_nm = np.asarray(mb_nm, dtype=self.neural_map.dtype)
                mb_nm_xy = np.asarray(mb_nm_xy, dtype=self.neural_map_xy.dtype)

                # Prepare nm_xy
                for i in range(self.neural_map.shape[0]):
                    self.neural_map_xy[i,:] = self.neural_map[i, int(self.pos[i,1]//2), int(self.pos[i,0]//2), :]

            last_values = self.model.value(self.obs, S=self.neural_map, M=self.neural_map_xy)
        else:
            last_values = self.model.value(self.obs, S=self.states, M=self.dones)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values

        if self.use_nm_customization:
            return (*map(sf01, (mb_obs, mb_returns, mb_nm_xy, mb_actions, mb_values, mb_neglogpacs, mb_pos, mb_nm)), epinfos)
        else:
            return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
