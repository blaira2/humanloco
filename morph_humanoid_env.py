import numpy as np
import mujoco as mj
from collections import deque
from gymnasium import spaces
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv
import xml.etree.ElementTree as ET


def compute_start_height(params, base_height=None):
    """
    Compute correct starting torso height so feet rest on the floor.
    """
    if base_height is None:
        thigh = params["THIGH_LENGTH"]
        shin = params["SHIN_LENGTH"]
        foot = params["FOOT_RADIUS"]
        base_height = thigh + shin + foot

    safety = 0.14  # extra buffer so initial state avoids penetration
    return base_height + safety


def calculate_base_height_from_xml(xml_file, leg_side="right"):
    """
    Calculate the vertical base height from the torso to the foot using the XML.
    """
    try:
        tree = ET.parse(xml_file)
    except (FileNotFoundError, ET.ParseError):
        return None

    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        return None

    torso_body = worldbody.find("./body[@name='torso']")
    if torso_body is None:
        return None

    chain = [
        "lwaist",
        "pelvis",
        f"{leg_side}_thigh",
        f"{leg_side}_shin",
        f"{leg_side}_foot",
    ]

    base_height = 0.0
    current = torso_body
    for body_name in chain:
        next_body = None
        for child in current.findall("body"):
            if child.attrib.get("name") == body_name:
                next_body = child
                break
        if next_body is None:
            return None
        pos = next_body.attrib.get("pos", "0 0 0")
        pos_z = float(pos.split()[2])
        base_height += abs(pos_z)
        current = next_body

    foot_geom = current.find("geom")
    if foot_geom is not None:
        size = foot_geom.attrib.get("size")
        if size:
            radius = float(size.split()[0])
            base_height += radius

    return base_height


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
        xml_base_height = calculate_base_height_from_xml(xml_file)
        self.base_height = xml_base_height if xml_base_height is not None else 1
        print(f"calculated base height: {self.base_height} ")

        self.start_height = compute_start_height(morph_params, base_height=self.base_height)
        self._prev_com_distance = None

        # Store for debugging
        self._prev_action = None
        self._prev_qvel = None
        self._prev_vertical_potential = None
        self._prev_angular_potential = None
        self._prev_com_position = None
        self._prev_x_position = None
        self._prev_x_progress = None

        # -----------------------------
        # Compute robust healthy_z_range
        # -----------------------------
        min_healthy = 0.25 * self.base_height
        max_healthy = 2.0 * self.base_height

        self.custom_healthy_min = min_healthy
        self.custom_healthy_max = max_healthy

        # stay healthy
        self._steps_alive = 0  # counts how many steps we stay alive

        self.prev_com_margin = 0
        self._phase_step = 0
        self.phase_cycle = 200
        self.vertical_velocity_shaping_weight = 0.5
        self.vertical_velocity_shaping_gamma = 0.99
        self.angular_velocity_shaping_weight = 0.5
        self.angular_velocity_shaping_gamma = 0.99
        self._com_x_vel_history = deque(maxlen=5)
        self._com_y_vel_history = deque(maxlen=5)
        self._com_z_vel_history = deque(maxlen=5)

        super().__init__(
            xml_file=xml_file,
            forward_reward_weight=0.0,
            terminate_when_unhealthy=True,
            **kwargs,
        )
        base_space = self.observation_space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(base_space.shape[0] + 1,),
            dtype=base_space.dtype,
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
        self.max_tilt = np.deg2rad(90)  # if torso tilts below horizontal, consider fallen
        self.min_base_height = 0.03  # if the pelvis hits the floor → terminal
        self._ground_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "floor")
        self._allowed_contact_geom_ids = {
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "left_foot"),
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "right_foot"),
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "left_hand"),
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "right_hand"),
        }

    def step(self, action):
        # ---- call original HumanoidEnv step ----
        obs, base_reward, terminated, truncated, info = super().step(action)
        self._phase_step += 1
        obs = self._get_obs()

        #Reward weights
        forward_reward_amount = 5
        com_alignment_weight = .5
        max_com_progress_weight = 0.75
        energy_weight = .8
        collision_weight = .1
        velocity_stability_weight = 2
        velocity_stability_deadzone = 0.05
        max_alive = -.5

        # Base kinematics
        x_vel = float(self.data.qvel[0])  # forward speed
        y_vel = float(self.data.qvel[1])  # lateral speed
        z_vel = float(self.data.qvel[2])  # downward speed
        ang_vel = self.data.qvel[3:6]  # angular vel


        # Alive reward
        # small constant per timestep + big penalty on fall
        self._steps_alive += 1
        alive_reward =  max_alive if not (terminated or truncated) else 0.0
        x_position = float(self.data.qpos[0])
        if self._prev_x_position is None:
            x_progress = 0.0
        else:
            x_progress = x_position - self._prev_x_position
        if (
            alive_reward != 0.0
            and self._prev_x_progress is not None
            and x_progress <= self._prev_x_progress
        ):
            alive_reward *= 4
        self._prev_x_position = x_position
        self._prev_x_progress = x_progress
        # terminal penalty shrinks over time
        terminal_penalty = 500
        if not terminated:  # only if it actually fell, not time-limit
            terminal_penalty = 0.0

        # Forward reward
        # set reward per step when forward COM velocity dominates and exceeds minimum
        com_position = np.array(self.data.subtree_com[0], dtype=float)
        if self._prev_com_position is None:
            com_velocity = np.zeros(3, dtype=float)
        else:
            com_velocity = (com_position - self._prev_com_position) / self.dt
        self._prev_com_position = com_position

        forward_speed = max(0.0, float(com_velocity[0]))
        self._com_x_vel_history.append(forward_speed)
        self._com_y_vel_history.append(abs(float(com_velocity[1])))
        self._com_z_vel_history.append(abs(float(com_velocity[2])))

        avg_forward_speed = float(np.mean(self._com_x_vel_history))
        avg_lateral_speed = float(np.mean(self._com_y_vel_history))
        avg_vertical_speed = float(np.mean(self._com_z_vel_history))
        max_other_speed = max(avg_lateral_speed, avg_vertical_speed)

        min_forward_speed = 0.1
        meets_forward_criteria = avg_forward_speed > min_forward_speed and avg_forward_speed > 2.0 * max_other_speed
        forward_reward = forward_reward_amount if meets_forward_criteria else 0.0

        #Velocity history stability
        velocity_delta = abs(x_vel - avg_forward_speed)
        velocity_stability_penalty = velocity_stability_weight * max(
            0.0, velocity_delta - velocity_stability_deadzone
        )

        # energy penalty = discourage huge torques
        action = np.asarray(action)
        energy = np.sum(action**2)
        n = len(action)
        energy_penalty = energy_weight * (energy / n)

        # contact penalty
        contact_penalty = 0.0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 == self._ground_geom_id:
                other = contact.geom2
            elif contact.geom2 == self._ground_geom_id:
                other = contact.geom1
            else:
                continue
            if other in self._allowed_contact_geom_ids:
                continue
            contact_force = np.zeros(6, dtype=float)
            mj.mj_contactForce(self.model, self.data, i, contact_force)
            normal_force = abs(contact_force[0])
            contact_penalty += collision_weight * normal_force

        # COM reward
        forward_scale = 1
        lateral_scale = .6
        forward_bias = .03
        torso_body_id = self.model.body("torso").id
        left_foot_id = self.model.body("left_foot").id
        right_foot_id = self.model.body("right_foot").id
        left_hand_geom_id = self.model.geom("left_hand").id
        right_hand_geom_id = self.model.geom("right_hand").id
        torso_xy = self.data.xipos[torso_body_id][:2]
        lf_xy = self.data.xipos[left_foot_id][:2]
        rf_xy = self.data.xipos[right_foot_id][:2]
        lh_xy = self.data.geom_xpos[left_hand_geom_id][:2]
        rh_xy = self.data.geom_xpos[right_hand_geom_id][:2]
        support_points = np.array([lf_xy, rf_xy, lh_xy, rh_xy], dtype=float)
        x_limits = (
            float(support_points[:, 0].min()) + forward_bias,
            float(support_points[:, 0].max()) + forward_bias,
        )
        y_limits = (float(support_points[:, 1].min()), float(support_points[:, 1].max()))
        support_center = np.array(
            [(x_limits[0] + x_limits[1]) / 2.0, (y_limits[0] + y_limits[1]) / 2.0],
            dtype=float,
        )
        com_xy = self.data.subtree_com[0][:2]
        dx = com_xy[0] - support_center[0]
        dy = com_xy[1] - support_center[1]
        weighted_distance = np.sqrt((dx / forward_scale) ** 2 + (dy / lateral_scale) ** 2)

        dx_outside = max(x_limits[0] - com_xy[0], 0.0, com_xy[0] - x_limits[1])
        dy_outside = max(y_limits[0] - com_xy[1], 0.0, com_xy[1] - y_limits[1])
        com_outside_distance = float(np.hypot(dx_outside, dy_outside))
        com_alignment_reward = com_alignment_weight * (1.0 - com_outside_distance)

        phase = (self._phase_step % self.phase_cycle) / self.phase_cycle
        com_progress_weight = max_com_progress_weight * (np.sin(2 * np.pi * phase) ** 2)

        if self._prev_com_distance is None:
            com_progress_reward = 0.0
        else:
            com_progress_reward = (com_progress_weight * (self._prev_com_distance - weighted_distance))
        self._prev_com_distance = weighted_distance

        com_alignment_reward += com_progress_reward

        # -----------------
        # Potential based shaping
        # -----------------

        # Potential-based shaping on vertical velocity
        vertical_potential = -abs(z_vel)
        if self._prev_vertical_potential is None:
            vertical_velocity_shaping = 0.0
        else:
            vertical_velocity_shaping = self.vertical_velocity_shaping_weight * (
                self.vertical_velocity_shaping_gamma * vertical_potential - self._prev_vertical_potential
            )
        self._prev_vertical_potential = vertical_potential

        # Potential-based shaping on angular velocity
        angular_speed = np.linalg.norm(ang_vel[:2])
        angular_potential = -angular_speed
        if self._prev_angular_potential is None:
            angular_velocity_shaping = 0.0
        else:
            angular_velocity_shaping = self.angular_velocity_shaping_weight * (
                self.angular_velocity_shaping_gamma * angular_potential - self._prev_angular_potential
            )
        self._prev_angular_potential = angular_potential




        # Combine

        reward = (
            alive_reward
            + forward_reward
            + com_alignment_reward
            + vertical_velocity_shaping
            + angular_velocity_shaping
            - terminal_penalty
            - energy_penalty
            - contact_penalty
            - velocity_stability_penalty
        )

        info["energy_penalty"] = float(energy_penalty)
        info["forward_reward"] = float(forward_reward)
        info["com_reward"] = float(com_alignment_reward)
        info["vertical_velocity_shaping"] = float(vertical_velocity_shaping)
        info["angular_velocity_shaping"] = float(angular_velocity_shaping)
        info["angular_speed"] = float(angular_speed)
        info["alive_reward"] = float(alive_reward)
        info["x_position"] = float(self.data.qpos[0])
        info["collision_penalty"] = float(contact_penalty)
        info["velocity_stability_penalty"] = float(velocity_stability_penalty)
        info["avg_forward_com_speed"] = float(avg_forward_speed)
        info["avg_lateral_com_speed"] = float(avg_lateral_speed)
        info["avg_vertical_com_speed"] = float(avg_vertical_speed)
        info["meets_forward_criteria"] = bool(meets_forward_criteria)

        return obs, reward, terminated, truncated, info

    # -----------------------------------
    # RESET
    # -----------------------------------
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)

        self._steps_alive = 0
        self._prev_action = None
        self._prev_qvel = None
        self._prev_com_distance = None
        self._prev_vertical_potential = None
        self._prev_angular_potential = None
        self._phase_step = 0
        self._prev_com_position = None
        self._prev_x_position = None
        self._prev_x_progress = None
        self._com_x_vel_history.clear()
        self._com_y_vel_history.clear()
        self._com_z_vel_history.clear()
        # If initial reset is below threshold, lift robot a little
        torso_z = obs[0]
        if torso_z < self.custom_healthy_min:
            delta = self.custom_healthy_min - torso_z + 0.05
            self.data.qpos[2] += delta
            mj.mj_forward(self.model, self.data)
        obs = self._get_obs()

        return obs, info

    def _get_obs(self):
        obs = super()._get_obs()
        phase = (self._phase_step % self.phase_cycle) / self.phase_cycle
        return np.concatenate([obs, np.array([phase], dtype=obs.dtype)])

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
