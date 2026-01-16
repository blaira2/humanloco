import numpy as np
from gymnasium import spaces
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
        velocity_penalty_weight=0.02,
        energy_penalty_weight=0.05,
        angular_velocity_penalty_weight=0.06,
        com_alignment_weight=1,
        upright_reward_weight=0.01,
        com_progress_weight=0.5,
        morph_params=None,
        **kwargs,
    ):
        self.velocity_penalty_weight = float(velocity_penalty_weight)
        self.energy_penalty_weight = float(energy_penalty_weight)
        self.angular_velocity_penalty_weight = float(
            angular_velocity_penalty_weight
        )
        self.com_alignment_weight = float(com_alignment_weight)
        self.upright_reward_weight = float(upright_reward_weight)
        self.com_progress_weight = float(com_progress_weight)
        self._steps_alive = 0
        self._prev_com_distance = None
        self._phase_step = 0
        self.phase_cycle = 200
        self.morph = morph_params

        super().__init__(
            xml_file=xml_file,
            forward_reward_weight=0.0,
            ctrl_cost_weight=0.0,
            contact_cost_weight=0.0,
            **kwargs,
        )
        base_space = self.observation_space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(base_space.shape[0] + 1,),
            dtype=base_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._steps_alive = 0
        self._prev_com_distance = None
        self._phase_step = 0
        info["morph_params"] = self.morph
        obs = self._get_obs()
        return obs, info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = super().step(action)
        self._phase_step += 1
        obs = self._get_obs()

        alive_step = 1
        self._steps_alive += 1
        alive_reward = alive_step if not (terminated or truncated) else 0.0
        # terminal penalty shrinks over time
        survival_frac = np.clip(self._steps_alive / 1000, 0.0, 1.0)
        max_penalty = 100.0
        terminal_penalty = max_penalty * (1.0 - survival_frac)
        if not terminated:  # only if it actually fell, not time-limit
            terminal_penalty = 0.0

        #velocity penalties
        root_lin_vel = np.asarray(self.data.qvel[0:3], dtype=float)
        non_forward_components = np.array(
            [max(0.0, -root_lin_vel[0]), root_lin_vel[1], root_lin_vel[2]],
            dtype=float,
        )
        non_forward_speed = np.linalg.norm(non_forward_components)
        velocity_penalty = self.velocity_penalty_weight * non_forward_speed
        root_ang_vel = np.asarray(self.data.qvel[3:6], dtype=float)
        angular_velocity_penalty = (
            self.angular_velocity_penalty_weight * np.linalg.norm(root_ang_vel)
        )

        action = np.asarray(action)
        energy_penalty = self.energy_penalty_weight * (np.sum(action**2) / len(action))

        torso_quat = self.data.xquat[1]
        w = float(torso_quat[0])
        tilt_angle = 2 * np.arccos(np.clip(abs(w), 0.0, 1.0))
        upright_reward = self.upright_reward_weight * np.exp(-tilt_angle)

        #COM reward
        torso_body_id = self.model.body("torso").id
        left_foot_id = self.model.body("left_foot").id
        right_foot_id = self.model.body("right_foot").id
        torso_xy = self.data.xipos[torso_body_id][:2]
        lf_xy = self.data.xipos[left_foot_id][:2]
        rf_xy = self.data.xipos[right_foot_id][:2]
        x_limits = (min(lf_xy[0], rf_xy[0]), max(lf_xy[0], rf_xy[0]))
        y_limits = (min(lf_xy[1], rf_xy[1]), max(lf_xy[1], rf_xy[1]))
        support_center = np.array(
            [(x_limits[0] + x_limits[1]) / 2.0, (y_limits[0] + y_limits[1]) / 2.0],
            dtype=float,
        )
        com_xy = self.data.subtree_com[0][:2]
        com_distance = float(np.linalg.norm(com_xy - support_center))
        dx_outside = max(x_limits[0] - com_xy[0], 0.0, com_xy[0] - x_limits[1])
        dy_outside = max(y_limits[0] - com_xy[1], 0.0, com_xy[1] - y_limits[1])
        com_outside_distance = float(np.hypot(dx_outside, dy_outside))
        com_alignment_reward = self.com_alignment_weight * (1.0 - com_outside_distance)
        
        if self._prev_com_distance is None:
            com_progress_reward = 0.0
        else:
            com_progress_reward = (self.com_progress_weight * (self._prev_com_distance - com_distance ))
        self._prev_com_distance = com_distance

        com_alignment_reward += com_progress_reward


        reward = (
            alive_reward
            + com_alignment_reward
            + upright_reward
            - velocity_penalty
            - angular_velocity_penalty
            - terminal_penalty
            - energy_penalty
        )

        info["alive_reward"] = float(alive_reward)
        info["velocity_penalty"] = float(velocity_penalty)
        info["energy_penalty"] = float(energy_penalty)
        info["com_reward"] = float(com_alignment_reward)
        info["angular_penalty"] = float(angular_velocity_penalty)
        info["upright_reward"] = float(upright_reward)
        info["morph_params"] = self.morph

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        obs = super()._get_obs()
        phase = (self._phase_step % self.phase_cycle) / self.phase_cycle
        return np.concatenate([obs, np.array([phase], dtype=obs.dtype)])
