import numpy as np
import mujoco as mj
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv


def compute_start_height(params):
    """
    Compute correct starting torso height so feet rest on the floor.
    """
    thigh = params["THIGH_LENGTH"]
    shin = params["SHIN_LENGTH"]
    foot = params["FOOT_RADIUS"]

    safety = 0.14  # extra buffer so initial state avoids penetration
    return thigh + shin + foot + safety


class MorphHumanoidEnv(HumanoidEnv):
    """
    Custom Humanoid environment that:
      • Adjusts healthy_z_range based on morphology
      • Uses a more robust fall detection method
      • Avoids instant terminal states for crouching or short robots
      • Still terminates properly when falling / lying down
    """

    def __init__(self, xml_file, morph_params, **kwargs):
        self.morph = morph_params

        # -----------------------------
        # Compute morphology-based torso height
        # -----------------------------
        thigh = morph_params.get("THIGH_LENGTH", 0.34)
        shin = morph_params.get("SHIN_LENGTH", 0.30)
        torso = morph_params.get("TORSO_LENGTH", 0.70)
        head = morph_params.get("HEAD_HEIGHT", 0.45)
        foot = morph_params.get("FOOT_RADIUS", 0.075)

        self.base_height = thigh + shin + foot  # hip -> ground distance
        upper_height = torso + head  # hip -> head distance
        standing_height = self.base_height + upper_height
        self.start_height = compute_start_height(morph_params)

        # Store for debugging
        self.morph_standing_height = standing_height
        self._prev_action = None
        self._prev_qvel = None
        # -----------------------------
        # Compute robust healthy_z_range
        # -----------------------------
        # Instead of:
        #   healthy if standing_height * 0.5 < z < standing_height * 1.5
        #
        # Use:
        #   • Torso must be ABOVE 20% of standing height
        #   • Torso must be BELOW 200% (sanity)
        #   • Allows crouching/kneeling but not lying flat
        #
        min_healthy = 0.15 * standing_height
        max_healthy = 2.0 * standing_height

        self.custom_healthy_min = min_healthy
        self.custom_healthy_max = max_healthy

        # stay healthy
        self._steps_alive = 0  # counts how many steps we stay alive

        self.forward_scale = 0.25
        self.prev_com_margin = 0

        super().__init__(
            xml_file=xml_file,
            forward_reward_weight=0.0,
            terminate_when_unhealthy=True,
            **kwargs,
        )

        # cache some body parts
        self.torso_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "torso1")
        self.head_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "head")
        self.torso_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "torso")
        self.left_foot_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "left_foot")
        self.right_foot_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "right_foot")

        self.init_qpos[2] = self.start_height

        # Override the internal range the parent env will check
        self._healthy_z_range = (self.custom_healthy_min, self.custom_healthy_max)

        # -----------------------------
        # 3. Additional robustness parameters
        # -----------------------------
        self.max_tilt = np.deg2rad(70)  # if torso tilts >70°, consider fallen
        self.min_base_height = 0.03  # if the pelvis hits the floor → terminal

    def step(self, action):
        # ---- call original HumanoidEnv step ----
        obs, base_reward, terminated, truncated, info = super().step(action)

        # ----------------------
        # Base kinematics
        # ----------------------
        x_vel = float(self.data.qvel[0])  # forward speed
        y_vel = float(self.data.qvel[1])  # lateral speed
        z_vel = float(self.data.qvel[2])  # downward speed
        x_vel = max(x_vel, 0.0)  # no reward for walking backwards
        ang_vel = self.data.qvel[3:6]  # angular vel

        # Alive reward
        # small constant per timestep + big penalty on fall
        alive_step = 1
        self._steps_alive += 1
        alive_reward = alive_step if not (terminated or truncated) else 0.0
        # terminal penalty shrinks over time
        survival_frac = np.clip(self._steps_alive / 1000, 0.0, 1.0)
        max_penalty = 140.0
        terminal_penalty = max_penalty * (1.0 - survival_frac)
        if not terminated:  # only if it actually fell, not time-limit
            terminal_penalty = 0.0

        # Forward reward
        # reward grows with speed but saturates, and is posture-gated
        x_vel_clipped = min(x_vel, 3.0)  # cap speed for reward purposes
        forward_base = self.forward_scale * x_vel_clipped

        # energy penalty = discourage huge torques
        action = np.asarray(action)
        energy = np.sum(action**2)
        n = len(action)
        energy_penalty = 1 * (energy / n)

        forward_reward = 5 * forward_base

        # -----------------
        # Minor Penalties
        # -----------------
        # acceleration penalty (penalize every direction but forward)
        accel_penalty = 0.0
        if self._prev_qvel is not None:
            dt = self.model.opt.timestep
            accel = (self.data.qvel[:3] - self._prev_qvel[:3]) / dt
            forward_axis = np.array([1.0, 0.0, 0.0])
            forward_accel = max(0.0, float(np.dot(accel, forward_axis)))
            non_forward_accel = accel - forward_accel * forward_axis
            accel_penalty = 0.05 * (1 - self.forward_scale) * np.linalg.norm(non_forward_accel)
        self._prev_qvel = self.data.qvel.copy()

        # sideways = discourage strafing
        lateral_penalty = 0.25 * abs(y_vel)

        # --- Center of mass and feet geometry ---
        com = np.array(self.data.subtree_com[self.torso_body_id])  # [x, y, z]
        com_xy = com[:2]
        lf_xy = self.data.xipos[self.left_foot_body_id, :2]
        rf_xy = self.data.xipos[self.right_foot_body_id, :2]
        feet_mid = 0.5 * (lf_xy + rf_xy)
        # is com near any balance point
        d_left = np.linalg.norm(com_xy - lf_xy)
        d_right = np.linalg.norm(com_xy - rf_xy)
        d_mid = np.linalg.norm(com_xy - feet_mid)
        d_min = float(min(d_left, d_right, d_mid))
        d_excess = max(1e-4, d_min - 0.1)
        d_min_dt = (d_excess - self.prev_com_margin) / self.model.opt.timestep
        self.prev_com_margin = d_excess
        com_penalty = 0.3 * d_min_dt
        if com_penalty < 0:
            com_penalty = 0.75 * com_penalty  # weaker reward than penalty

        # angular velocity penalty
        angular_penalty = 0.05 * np.linalg.norm(ang_vel[:2])  # pitch/roll wobble

        # Combine

        reward = (
            alive_reward
            + forward_reward
            - terminal_penalty
            - lateral_penalty
            - accel_penalty
            - angular_penalty
            - energy_penalty
            - com_penalty
        )

        info["energy_penalty"] = float(energy_penalty)
        info["forward_reward"] = float(forward_reward)
        info["lateral_penalty"] = float(lateral_penalty)
        info["accel_penalty"] = float(accel_penalty)
        info["com_penalty"] = float(com_penalty)
        info["angular_penalty"] = float(angular_penalty)
        info["alive_reward"] = float(alive_reward)

        return obs, reward, terminated, truncated, info

    # -----------------------------------
    # RESET
    # -----------------------------------
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)

        self._steps_alive = 0
        self._prev_action = None
        self._prev_qvel = None
        # If initial reset is below threshold, lift robot a little
        torso_z = obs[0]
        if torso_z < self.custom_healthy_min:
            delta = self.custom_healthy_min - torso_z + 0.05
            self.data.qpos[2] += delta
            mj.mj_forward(self.model, self.data)
            obs = self._get_obs()

        return obs, info

    # -----------------------------------
    # CUSTOM HEALTH CHECK
    # -----------------------------------
    def _is_healthy(self, state):
        """
        Improved fall detection:
          • Torso above minimum height
          • Torso below maximum height
          • Torso not excessively tilted
          • Pelvis not touching floor
        """

        torso_z = state[2]
        min_z, max_z = self._healthy_z_range

        # Height check
        if not (min_z <= torso_z <= max_z):
            return False

        # Pelvis height (qpos index 2 is torso, index 5 is pelvis if using Humanoid-v5 defaults)
        pelvis_z = float(self.data.qpos[2])
        if pelvis_z < self.min_base_height:
            return False

        # Torso tilt check
        # Orientation is quaternion (w, x, y, z), Humanoid-v5 places torso orientation early in qpos
        torso_quat = self.data.xquat[1]  # body index 1 = torso in default humanoid
        w, x, y, z = torso_quat

        # Convert quaternion → tilt angle
        tilt_angle = 2 * np.arccos(abs(w))
        if tilt_angle > self.max_tilt:
            return False

        return True

    def set_forward_scale(self, scale: float):
        self.forward_scale = float(scale)

    # -----------------------------------
    # Optional diagnostics helper
    # -----------------------------------
    def print_diagnostics(self, state):
        z = state[2]
        min_z, max_z = self._healthy_z_range

        print("\n--- HEALTH DIAGNOSTICS ---")
        print(f"Standing height:       {self.morph_standing_height:.3f}")
        print(f"Healthy z-range:       [{min_z:.3f}, {max_z:.3f}]")
        print(f"Current torso z:       {z:.3f}")
        print(f"Base pelvis height:    {float(self.data.qpos[2]):.3f}")

        # tilt
        w, x, y, zq = self.data.xquat[1]
        tilt = 2 * np.arccos(abs(w))
        print(f"Torso tilt angle deg:  {np.rad2deg(tilt):.2f}")
        print("--------------------------")
