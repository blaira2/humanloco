import numpy as np
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv


class BalanceHumanoidEnv(HumanoidEnv):
    """
    Humanoid environment focused on balance.

    Reward is based on staying alive, with a penalty for downward acceleration.
    """

    def __init__(
        self,
        xml_file=None,
        downward_accel_weight=1.0,
        energy_penalty_weight=0.01,
        morph_params=None,
        **kwargs,
    ):
        self.downward_accel_weight = float(downward_accel_weight)
        self.energy_penalty_weight = float(energy_penalty_weight)
        self._prev_z_vel = None
        self._steps_alive = 0
        self.morph = morph_params

        super().__init__(
            xml_file=xml_file,
            forward_reward_weight=0.0,
            ctrl_cost_weight=0.0,
            contact_cost_weight=0.0,
            **kwargs,
        )

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._prev_z_vel = None
        self._steps_alive = 0
        info["morph_params"] = self.morph
        return obs, info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = super().step(action)

        alive_step = 1
        self._steps_alive += 1
        alive_reward = alive_step if not (terminated or truncated) else 0.0
        # terminal penalty shrinks over time
        survival_frac = np.clip(self._steps_alive / 1000, 0.0, 1.0)
        max_penalty = 120.0
        terminal_penalty = max_penalty * (1.0 - survival_frac)
        if not terminated:  # only if it actually fell, not time-limit
            terminal_penalty = 0.0

        downward_accel_penalty = 0.0
        if self._prev_z_vel is not None:
            dt = self.model.opt.timestep
            accel_z = (self.data.qvel[2] - self._prev_z_vel) / dt
            downward_accel = max(0.0, -float(accel_z))
            downward_accel_penalty = self.downward_accel_weight * downward_accel

        self._prev_z_vel = float(self.data.qvel[2])

        action = np.asarray(action)
        energy_penalty = self.energy_penalty_weight * (np.sum(action**2) / len(action))

        reward = (
            alive_reward
            - downward_accel_penalty
            - terminal_penalty
            - energy_penalty
        )

        info["alive_reward"] = float(alive_reward)
        info["accel_penalty"] = float(downward_accel_penalty)
        info["energy_penalty"] = float(energy_penalty)
        info["morph_params"] = self.morph

        return obs, reward, terminated, truncated, info
