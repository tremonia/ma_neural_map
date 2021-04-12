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
    def __init__(self, env, model, nsteps=5, gamma=0.99, use_extended_write_op=False, max_positions = [10, 10]):
        super().__init__(env=env, model=model, nsteps=nsteps,
                         use_extended_write_op=use_extended_write_op, max_positions=max_positions)
        self.gamma = gamma
        self.batch_action_shape = [x if x is not None else -1 for x in model.train_model.action.shape.as_list()]
        self.ob_dtype = model.train_model.X.dtype.as_numpy_dtype

    def run(self):
        # We initialize the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_pos, mb_nm, mb_nm_xy = [],[],[],[],[],[],[],[]
        epinfos = []
        for n in range(self.nsteps):
            # Given observations, take action and value (V(s))
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init


                else:

            # Append the experiences
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)

            mb_pos.append(self.pos.copy())
            mb_nm.append(self.neural_map.copy())
            mb_nm_xy.append(self.neural_map_xy.copy())


                else:
                    self.neural_map[i, int(self.pos[i,1] // self.pos_y_divisor), int(self.pos[i,0] // self.pos_x_divisor), :] = write_vector[i,:]

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            tmp, rewards, dones, infos = self.env.step(actions)
            obs = tmp[:,:-3]
            self.pos = tmp[:,-3:]

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)

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
        mb_pos = np.asarray(mb_pos, dtype=self.pos.dtype)
        mb_nm = np.asarray(mb_nm, dtype=self.neural_map.dtype)
        mb_nm = mb_nm.swapaxes(0, 1).reshape(mb_nm.shape[0] * mb_nm.shape[1], *mb_nm.shape[2:])
        mb_nm_xy = np.asarray(mb_nm_xy, dtype=self.neural_map_xy.dtype)
        mb_nm_xy = mb_nm_xy.swapaxes(0, 1).reshape(mb_nm_xy.shape[0] * mb_nm_xy.shape[1], *mb_nm_xy.shape[2:])
        if self.gamma > 0.0:
            # Discount/bootstrap off value fn
                else:

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

        return mb_obs, mb_nm, mb_rewards, mb_nm_xy, mb_actions, mb_values, epinfos
