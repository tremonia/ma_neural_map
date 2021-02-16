import numpy as np
from baselines.a2c.utils import discount_with_dones
from baselines.common.runners import AbstractEnvRunner

class Runner(AbstractEnvRunner):
    """
    We use this class to generate batches of experiences

    __init__:
    - Initialize the runner

    run():
    - Make a mini batch of experiences
    """
    def __init__(self, env, model, nsteps=5, gamma=0.99, use_nm_customization=False):
        super().__init__(env=env, model=model, nsteps=nsteps, use_nm_customization=use_nm_customization)
        self.gamma = gamma
        self.batch_action_shape = [x if x is not None else -1 for x in model.train_model.action.shape.as_list()]
        self.ob_dtype = model.train_model.X.dtype.as_numpy_dtype

    def run(self):
        # We initialize the lists that will contain the mb of experiences
        if self.use_nm_customization:
            mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_pos, mb_nm, mb_nm_xy = [],[],[],[],[],[],[],[]
        else:
            mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        mb_states = self.states
        epinfos = []
        for n in range(self.nsteps):
            # Given observations, take action and value (V(s))
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init

            if self.use_nm_customization:
                if self.model.initial_state is not None:
                    # Prepare nm_xy
                    for i in range(self.neural_map.shape[0]):
                        self.neural_map_xy[i,:] = self.neural_map[i, int(self.pos[i,1]//2), int(self.pos[i,0]//2), :]

                    actions, values, write_vector, _ = self.model.step(self.obs, S=self.neural_map, M=self.neural_map_xy)
                else:
                    # Non-recurrent model, i.e. NOT neural map
                    actions, values, states, _ = self.model.step(self.obs, S=None, M=None)
            else:
                actions, values, states, _ = self.model.step(self.obs, S=self.states, M=self.dones)

            # Append the experiences
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
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
                tmp, rewards, dones, infos = self.env.step(actions)
                obs = tmp[:,:-2]
                self.pos = tmp[:,-2:]
            else:
                obs, rewards, dones, infos = self.env.step(actions)

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            #self.states = states TO DO: FIX THIS LINE IN CASE OF NEURAL MAP IS USED !!!!!!!!!!!!!!!!!!!!!!!!!!!
            self.dones = dones
            self.obs = obs
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)

        # Batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.ob_dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=self.model.train_model.action.dtype.name).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]


        if self.gamma > 0.0:
            # Discount/bootstrap off value fn
            if self.use_nm_customization:
                mb_pos = np.asarray(mb_pos, dtype=self.pos.dtype)
                if self.model.initial_state is not None:
                    mb_nm = np.asarray(mb_nm, dtype=self.neural_map.dtype)
                    mb_nm = mb_nm.swapaxes(0, 1).reshape(mb_nm.shape[0] * mb_nm.shape[1], *mb_nm.shape[2:])
                    mb_nm_xy = np.asarray(mb_nm_xy, dtype=self.neural_map_xy.dtype)
                    mb_nm_xy = mb_nm_xy.swapaxes(0, 1).reshape(mb_nm_xy.shape[0] * mb_nm_xy.shape[1], *mb_nm_xy.shape[2:])

                    # Prepare nm_xy
                    for i in range(self.neural_map.shape[0]):
                        self.neural_map_xy[i,:] = self.neural_map[i, int(self.pos[i,1]//2), int(self.pos[i,0]//2), :]

                    last_values = self.model.value(self.obs, S=self.neural_map, M=self.neural_map_xy).tolist()
                else:
                    # Non-recurrent model, i.e. NOT neural map
                    last_values = self.model.value(self.obs, S=None, M=None).tolist()
            else:
                last_values = self.model.value(self.obs, S=self.states, M=self.dones).tolist()

            for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
                rewards = rewards.tolist()
                dones = dones.tolist()
                if dones[-1] == 0:
                    rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, dones, self.gamma)

                mb_rewards[n] = rewards

        mb_actions = mb_actions.reshape(self.batch_action_shape)

        mb_rewards = mb_rewards.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()

        if self.model.initial_state is not None:
            return mb_obs, mb_nm, mb_rewards, mb_nm_xy, mb_actions, mb_values, epinfos
        else:
            # Non-recurrent model, i.e. NOT neural map
            return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, epinfos
