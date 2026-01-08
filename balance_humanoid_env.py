import numpy as np
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv


class BalanceHumanoidEnv(HumanoidEnv):
    """
    Humanoid environment focused on balance.

    Reward is based on staying alive, with penalties for instability and a bonus for
    staying upright.
    """

    def __init__(
        self,
        xml_file=None,
        downward_accel_weight=0.015,
        energy_penalty_weight=0.04,
        angular_velocity_penalty_weight=0.04,
        com_alignment_weight=0.3,
        upright_reward_weight=0.4,
        morph_params=None,
        **kwargs,
    ):
        self.downward_accel_weight = float(downward_accel_weight)
        self.energy_penalty_weight = float(energy_penalty_weight)
        self.angular_velocity_penalty_weight = float(
            angular_velocity_penalty_weight
        )
        self.com_alignment_weight = float(com_alignment_weight)
        self.upright_reward_weight = float(upright_reward_weight)
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
        min_safe_accel = 1.0;
        if self._prev_z_vel is not None:
            dt = self.model.opt.timestep
            accel_z = (self.data.qvel[2] - self._prev_z_vel) / dt
            downward_accel = max(0.0, -float(accel_z))
            if downward_accel < min_safe_accel:
                downward_accel = 0
            downward_accel_penalty = self.downward_accel_weight * downward_accel

        self._prev_z_vel = float(self.data.qvel[2])

        action = np.asarray(action)
        energy_penalty = self.energy_penalty_weight * (np.sum(action**2) / len(action))

        angular_velocity_penalty = (
            self.angular_velocity_penalty_weight
            * float(np.linalg.norm(self.data.qvel[3:6]))
        )

        torso_quat = self.data.xquat[1]
        w = float(torso_quat[0])
        tilt_angle = 2 * np.arccos(np.clip(abs(w), 0.0, 1.0))
        upright_reward = self.upright_reward_weight * np.exp(-tilt_angle)

        torso_body_id = self.model.body("torso").id
        torso_xy = self.data.xipos[torso_body_id][:2]
        com_xy = self.data.subtree_com[0][:2]
        com_offset = float(np.linalg.norm(torso_xy - com_xy))
        com_alignment_reward = self.com_alignment_weight * np.exp(-com_offset)

        reward = (
            alive_reward
            - downward_accel_penalty
            - terminal_penalty
            - energy_penalty
            - angular_velocity_penalty
            + com_alignment_reward
            + upright_reward
        )

        info["alive_reward"] = float(alive_reward)
        info["accel_penalty"] = float(downward_accel_penalty)
        info["energy_penalty"] = float(energy_penalty)
        info["angular_penalty"] = float(angular_velocity_penalty)
        info["com_penalty"] = float(-com_alignment_reward)
        info["upright_reward"] = float(upright_reward)
        info["morph_params"] = self.morph

        return obs, reward, terminated, truncated, info
