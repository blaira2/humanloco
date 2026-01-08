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
        morph_params=None,
        **kwargs,
    ):
        self.downward_accel_weight = float(downward_accel_weight)
        self._prev_z_vel = None
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
        info["morph_params"] = self.morph
        return obs, info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = super().step(action)

        alive_reward = self.healthy_reward if not (terminated or truncated) else 0.0

        downward_accel_penalty = 0.0
        if self._prev_z_vel is not None:
            dt = self.model.opt.timestep
            accel_z = (self.data.qvel[2] - self._prev_z_vel) / dt
            downward_accel = max(0.0, -float(accel_z))
            downward_accel_penalty = self.downward_accel_weight * downward_accel

        self._prev_z_vel = float(self.data.qvel[2])

        reward = alive_reward - downward_accel_penalty

        info["alive_reward"] = float(alive_reward)
        info["downward_accel_penalty"] = float(downward_accel_penalty)
        info["morph_params"] = self.morph

        return obs, reward, terminated, truncated, info
