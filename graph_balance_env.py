import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv


class GraphBalanceHumanoidEnv(HumanoidEnv):
    """An Environment that emits graph-structured observations for GCN policies."""

    def __init__(
        self,
        xml_file=None,
        morph_params=None,
        node_feature_dim=8,
        global_feature_dim=32,
        reset_height_step=0.0025,
        reset_max_drop=0.2,
        com_safe_window_weight=2,
        com_safe_window_progress_weight=0.5,
        velocity_shaping_weight=0.5,
        velocity_shaping_gamma=0.99,
        angular_velocity_shaping_weight=0.5,
        angular_velocity_shaping_gamma=0.99,
        velocity_penalty_weight=0.02,
        energy_penalty_weight=0.05,
        angular_velocity_penalty_weight=0.06,
        com_alignment_weight=1,
        upright_reward_weight=0.01,
        com_progress_weight=0.5,
        upper_body_above_end_effectors_weight=1.0,
        angular_divergence_penalty_weight=1.0,
        min_tilt_failure_height_ratio=0.4,
        min_tilt_failure_height_floor=0.4,
        unhealthy_torso_height_ratio=0.15,
        **kwargs,
    ):
        self.node_feature_dim = int(node_feature_dim)
        self.global_feature_dim = int(global_feature_dim)
        self.reset_height_step = float(reset_height_step)
        self.reset_max_drop = float(reset_max_drop)
        self.com_safe_window_weight = float(com_safe_window_weight)
        self.com_safe_window_progress_weight = float(com_safe_window_progress_weight)
        self.velocity_shaping_weight = float(velocity_shaping_weight)
        self.velocity_shaping_gamma = float(velocity_shaping_gamma)
        self.velocity_penalty_weight = float(velocity_penalty_weight)
        self.angular_velocity_shaping_weight = float(angular_velocity_shaping_weight)
        self.angular_velocity_shaping_gamma = float(angular_velocity_shaping_gamma)
        self.graph_energy_penalty_weight = float(energy_penalty_weight)
        self.energy_penalty_weight = float(energy_penalty_weight)
        self.angular_velocity_penalty_weight = float(angular_velocity_penalty_weight)
        self.com_alignment_weight = float(com_alignment_weight)
        self.upright_reward_weight = float(upright_reward_weight)
        self.com_progress_weight = float(com_progress_weight)
        self.upper_body_above_end_effectors_weight = float(
            upper_body_above_end_effectors_weight
        )
        self.angular_divergence_penalty_weight = float(
            angular_divergence_penalty_weight
        )
        self._min_tilt_failure_height_ratio = float(min_tilt_failure_height_ratio)
        self._min_tilt_failure_height_floor = float(min_tilt_failure_height_floor)
        self._unhealthy_torso_height_ratio = float(unhealthy_torso_height_ratio)
        self._starting_torso_height = None
        self._use_morphology_aware_healthy_z_range = "healthy_z_range" not in kwargs
        self._prev_velocity_potential = None
        self._prev_angular_velocity_potential = None
        self._prev_com_window_distance = None
        self._prev_com_distance = None
        self._steps_alive = 0
        self._phase_step = 0
        self.phase_cycle = 200
        self.morph = morph_params

        super().__init__(
            xml_file=xml_file,
            forward_reward_weight=0.0,
            ctrl_cost_weight=0.0,
            contact_cost_weight=0.0,
            healthy_reward=0.0,
            **kwargs,
        )

        base_space = self.observation_space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(base_space.shape[0] + 1,),
            dtype=base_space.dtype,
        )

        self._num_nodes = int(self.model.nbody - 1)  # skip world body
        self._adjacency = self._build_adjacency().astype(np.float32)
        self._limb_end_body_ids = self._find_limb_end_body_ids()
        self._limb_end_viz_body_ids, self._limb_end_geom_ids = self._find_limb_end_geom_ids()
        self._leaf_node_rgba = np.array([0.0, 0.1, 1.0, 1.0], dtype=float)
        self._torso_safe_rgba = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
        self._torso_alert_rgba = np.array([1.0, 0.0, 0.0, 1.0], dtype=float)
        self._torso_geom_id = self._find_torso_geom_id()
        self._unhealthy_ground_geom_ids = self._find_ground_geom_ids()
        self._unhealthy_body_geom_ids = self._find_unhealthy_body_geom_ids()

        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._num_nodes, self.node_feature_dim),
                    dtype=np.float32,
                ),
                "adjacency": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self._num_nodes, self._num_nodes),
                    dtype=np.float32,
                ),
                "global_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.global_feature_dim,),
                    dtype=np.float32,
                ),
            }
        )

    def _build_adjacency(self):
        adjacency = np.eye(self._num_nodes, dtype=np.float32)
        for body_id in range(1, self.model.nbody):
            parent_id = int(self.model.body_parentid[body_id])
            if parent_id <= 0:
                continue
            i = body_id - 1
            j = parent_id - 1
            adjacency[i, j] = 1.0
            adjacency[j, i] = 1.0
        return adjacency

    def _find_limb_end_body_ids(self):
        """Return graph leaf bodies (limb ends), excluding world and root torso."""
        child_counts = np.zeros(self.model.nbody, dtype=np.int32)
        for body_id in range(1, self.model.nbody):
            parent_id = int(self.model.body_parentid[body_id])
            if parent_id >= 0:
                child_counts[parent_id] += 1

        # Leaf nodes in the kinematic tree, ignoring world (0) and direct root (1).
        limb_end_ids = [
            body_id
            for body_id in range(2, self.model.nbody)
            if child_counts[body_id] == 0
        ]
        return tuple(limb_end_ids)

    def _find_limb_end_geom_ids(self):
        """Return geom ids associated with each limb-end body, in matching order."""
        body_ids = []
        geom_ids = []
        for body_id in self._limb_end_body_ids:
            geom_id = int(self.model.body_geomadr[body_id])
            geom_count = int(self.model.body_geomnum[body_id])
            if geom_count <= 0 or geom_id < 0:
                continue
            body_ids.append(body_id)
            geom_ids.append(geom_id)
        return tuple(body_ids), tuple(geom_ids)

    def _set_limb_end_geom_rgba(self, geom_id, rgba):
        self.model.geom_rgba[geom_id] = np.asarray(rgba, dtype=float)

    def _set_torso_geom_rgba(self, rgba):
        if self._torso_geom_id is None:
            return
        self.model.geom_rgba[self._torso_geom_id] = np.asarray(rgba, dtype=float)

    def _find_torso_geom_id(self):
        torso_body_id = self.model.body("torso").id
        geom_start = int(self.model.body_geomadr[torso_body_id])
        geom_count = int(self.model.body_geomnum[torso_body_id])
        if geom_count <= 0 or geom_start < 0:
            return None
        return geom_start

    def _find_ground_geom_ids(self):
        ground_names = ("floor", "ground")
        geom_ids = set()
        for geom_name in ground_names:
            try:
                geom_ids.add(int(self.model.geom(geom_name).id))
            except KeyError:
                continue
        return tuple(sorted(geom_ids))

    def _find_unhealthy_body_geom_ids(self):
        body_names = ("torso")
        geom_ids = []
        for body_name in body_names:
            try:
                body_id = self.model.body(body_name).id
            except KeyError:
                continue

            geom_start = int(self.model.body_geomadr[body_id])
            geom_count = int(self.model.body_geomnum[body_id])
            if geom_start < 0 or geom_count <= 0:
                continue

            for geom_id in range(geom_start, geom_start + geom_count):
                geom_ids.append(int(geom_id))

        return tuple(sorted(set(geom_ids)))

    def _has_unhealthy_ground_contact(self):
        if not self._unhealthy_ground_geom_ids or not self._unhealthy_body_geom_ids:
            return False

        ground_geom_ids = set(self._unhealthy_ground_geom_ids)
        body_geom_ids = set(self._unhealthy_body_geom_ids)

        for contact_idx in range(self.data.ncon):
            contact = self.data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)

            if (
                (geom1 in body_geom_ids and geom2 in ground_geom_ids)
                or (geom2 in body_geom_ids and geom1 in ground_geom_ids)
            ):
                return True

        return False

    def _is_torso_too_low(self):
        if self._starting_torso_height is None:
            return False

        torso_body_id = self.model.body("torso").id
        torso_height = float(self.data.xipos[torso_body_id][2])
        min_healthy_height = (
            self._unhealthy_torso_height_ratio * self._starting_torso_height
        )
        return torso_height < min_healthy_height

    @property
    def is_healthy(self):
        return not (self._has_unhealthy_ground_contact() or self._is_torso_too_low())

    @property
    def terminated(self):
        return (not self.is_healthy) if self._terminate_when_unhealthy else False

    def _reset_com_component_visualization(self):
        for geom_id in self._limb_end_geom_ids:
            self._set_limb_end_geom_rgba(geom_id, self._leaf_node_rgba)
        self._set_torso_geom_rgba(self._torso_safe_rgba)

    def _update_com_component_visualization(self, com_xy, reward_fraction):
        if not self._limb_end_geom_ids:
            return

        limb_end_xy = self.data.xipos[list(self._limb_end_viz_body_ids), :2]

        x_min = float(limb_end_xy[:, 0].min())
        x_max = float(limb_end_xy[:, 0].max())
        y_min = float(limb_end_xy[:, 1].min())
        y_max = float(limb_end_xy[:, 1].max())
        outside_distance = float(
            np.hypot(
                max(x_min - com_xy[0], 0.0, com_xy[0] - x_max),
                max(y_min - com_xy[1], 0.0, com_xy[1] - y_max),
            )
        )
        if outside_distance <= 0.0:
            blend_strength = float(np.clip(reward_fraction, 0.0, 1.0))
        else:
            blend_strength = 0.0
        torso_color = (
            (1.0 - blend_strength) * self._torso_alert_rgba
            + blend_strength * self._torso_safe_rgba
        )

        for geom_id in self._limb_end_geom_ids:
            self._set_limb_end_geom_rgba(geom_id, self._leaf_node_rgba)
        self._set_torso_geom_rgba(torso_color)

    def _com_safe_window_reward(self):
        """Reward CoM when it stays inside the box from outermost limb-end points."""
        if not self._limb_end_body_ids:
            return 0.0, False, 0.0

        limb_end_xy = self.data.xipos[list(self._limb_end_body_ids), :2]
        x_min = float(limb_end_xy[:, 0].min())
        x_max = float(limb_end_xy[:, 0].max())
        y_min = float(limb_end_xy[:, 1].min())
        y_max = float(limb_end_xy[:, 1].max())

        com_xy = self.data.subtree_com[0][:2]
        inside_x = x_min <= float(com_xy[0]) <= x_max
        inside_y = y_min <= float(com_xy[1]) <= y_max
        inside_window = inside_x and inside_y

        dx_outside = max(x_min - com_xy[0], 0.0, com_xy[0] - x_max)
        dy_outside = max(y_min - com_xy[1], 0.0, com_xy[1] - y_max)
        outside_distance = float(np.hypot(dx_outside, dy_outside))
        window_center = np.array(
            [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0],
            dtype=float,
        )
        com_window_distance = float(np.linalg.norm(com_xy - window_center))

        half_width = max((x_max - x_min) / 2.0, 1e-6)
        half_height = max((y_max - y_min) / 2.0, 1e-6)
        offset_x = abs(float(com_xy[0]) - float(window_center[0]))
        offset_y = abs(float(com_xy[1]) - float(window_center[1]))

        # Linear falloff from center of safe window; reaches 0 at window boundary.
        normalized_offset = max(offset_x / half_width, offset_y / half_height)
        center_alignment = float(np.clip(1.0 - normalized_offset, 0.0, 1.0))
        safe_window_reward = self.com_safe_window_weight * center_alignment

        if self._prev_com_window_distance is None:
            progress_reward = 0.0
        else:
            progress_reward = self.com_safe_window_progress_weight * (
                self._prev_com_window_distance - com_window_distance
            )
        self._prev_com_window_distance = com_window_distance

        safe_window_reward += progress_reward

        if self.com_safe_window_weight > 0.0:
            reward_fraction = center_alignment
        else:
            reward_fraction = float(inside_window)
        reward_fraction = float(np.clip(reward_fraction, 0.0, 1.0))
        self._update_com_component_visualization(com_xy, reward_fraction)

        return safe_window_reward, inside_window, outside_distance

    def _upper_body_above_end_effectors_reward(self):
        """Reward head + torso staying above the highest end-effector point."""
        if not self._limb_end_body_ids:
            return 0.0, 0.0

        torso_body_id = self.model.body("torso").id
        torso_z = float(self.data.xipos[torso_body_id][2])

        limb_end_z = self.data.xipos[list(self._limb_end_body_ids), 2]
        max_end_effector_z = float(np.max(limb_end_z))

        torso_margin = torso_z - max_end_effector_z

        torso_above_score = float(np.clip(torso_margin, 0.0, 1.0))
        mean_above_score = 0.5 * (torso_above_score )

        reward = self.upper_body_above_end_effectors_weight * mean_above_score
        return reward, torso_margin

    def _flat_to_graph_obs(self, flat_obs):
        flat_obs = np.asarray(flat_obs, dtype=np.float32)

        phase = np.array([flat_obs[-1]], dtype=np.float32)
        core = flat_obs[:-1]

        node_size = self._num_nodes * self.node_feature_dim
        if core.size >= node_size:
            node_flat = core[:node_size]
            rem = core[node_size:]
        else:
            node_flat = np.pad(core, (0, node_size - core.size))
            rem = np.zeros(0, dtype=np.float32)

        node_features = node_flat.reshape(self._num_nodes, self.node_feature_dim)

        global_raw = np.concatenate([rem, phase], dtype=np.float32)
        if global_raw.size >= self.global_feature_dim:
            global_features = global_raw[: self.global_feature_dim]
        else:
            global_features = np.pad(
                global_raw,
                (0, self.global_feature_dim - global_raw.size),
            )

        return {
            "node_features": node_features,
            "adjacency": self._adjacency,
            "global_features": global_features.astype(np.float32),
        }

    def _set_morphology_aware_healthy_z_range(self):
        if not self._use_morphology_aware_healthy_z_range:
            return

        # Humanoid-v5 fall/tilt termination uses healthy_z_range. Derive it
        # from post-adjustment torso height so reset lowering is accounted for.
        torso_z0 = float(self.data.qpos[2])
        ratio_min = self._min_tilt_failure_height_ratio * torso_z0
        floor_min = self._min_tilt_failure_height_floor
        healthy_min = min(torso_z0, max(floor_min, ratio_min))
        healthy_max = max(2.0, torso_z0 * 2.0)
        self._healthy_z_range = (healthy_min, healthy_max)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._prev_velocity_potential = None
        self._prev_angular_velocity_potential = None
        self._prev_com_window_distance = None
        self._prev_com_distance = None
        self._phase_step = 0
        self._lower_to_ground_contact_threshold()
        self._set_morphology_aware_healthy_z_range()
        torso_body_id = self.model.body("torso").id
        self._starting_torso_height = float(self.data.xipos[torso_body_id][2])
        self._steps_alive = 0
        self._reset_com_component_visualization()
        info["morph_params"] = self.morph
        obs = self._get_obs()
        return self._flat_to_graph_obs(obs), info

    def _lower_to_ground_contact_threshold(self):
        """Lower root body until floor contact would occur, then keep last safe pose."""
        if self.reset_height_step <= 0.0 or self.reset_max_drop <= 0.0:
            return

        try:
            floor_geom_id = self.model.geom("floor").id
        except KeyError:
            return

        base_qpos = self.data.qpos.copy()
        base_qvel = np.zeros_like(self.data.qvel)

        num_steps = int(np.floor(self.reset_max_drop / self.reset_height_step))
        last_safe_qpos = base_qpos.copy()

        for step_idx in range(1, num_steps + 1):
            candidate_qpos = base_qpos.copy()
            candidate_qpos[2] -= step_idx * self.reset_height_step
            self.set_state(candidate_qpos, base_qvel)

            has_floor_contact = False
            for contact_idx in range(self.data.ncon):
                contact = self.data.contact[contact_idx]
                if contact.geom1 == floor_geom_id or contact.geom2 == floor_geom_id:
                    has_floor_contact = True
                    break

            if has_floor_contact:
                break

            last_safe_qpos = candidate_qpos

        self.set_state(last_safe_qpos, base_qvel)

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        self._phase_step += 1
        obs = self._get_obs()

        self._steps_alive += 1

        # Balance reward terms (without alive reward).
        survival_frac = np.clip(self._steps_alive / 1000, 0.0, 1.0)
        max_penalty = 100.0
        terminal_penalty = max_penalty * (1.0 - survival_frac)
        if not terminated:  # only if it actually fell, not time-limit
            terminal_penalty = 0.0

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

        action = np.asarray(action, dtype=float)
        energy_penalty = self.energy_penalty_weight * float(np.mean(action**2))

        torso_quat = self.data.xquat[1]
        w = float(torso_quat[0])
        tilt_angle = 2 * np.arccos(np.clip(abs(w), 0.0, 1.0))
        upright_reward = self.upright_reward_weight * np.exp(-tilt_angle)

        torso_body_id = self.model.body("torso").id
        left_foot_id = self.model.body("left_foot").id
        right_foot_id = self.model.body("right_foot").id
        left_hand_geom_id = self.model.geom("left_hand").id
        right_hand_geom_id = self.model.geom("right_hand").id
        lf_xy = self.data.xipos[left_foot_id][:2]
        rf_xy = self.data.xipos[right_foot_id][:2]
        lh_xy = self.data.geom_xpos[left_hand_geom_id][:2]
        rh_xy = self.data.geom_xpos[right_hand_geom_id][:2]
        support_points = np.array([lf_xy, rf_xy, lh_xy, rh_xy], dtype=float)
        x_limits = (float(support_points[:, 0].min()), float(support_points[:, 0].max()))
        y_limits = (float(support_points[:, 1].min()), float(support_points[:, 1].max()))
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
            com_progress_reward = (
                self.com_progress_weight * (self._prev_com_distance - com_distance)
            )
        self._prev_com_distance = com_distance
        com_alignment_reward += com_progress_reward

        torso_rotation = self.data.xmat[torso_body_id].reshape(3, 3)
        torso_forward_world = torso_rotation[:, 0]
        world_forward = np.array([1.0, 0.0, 0.0], dtype=float)
        forward_alignment = float(
            np.clip(np.dot(torso_forward_world, world_forward), -1.0, 1.0)
        )
        torso_forward_divergence = float(np.arccos(forward_alignment))
        angular_divergence_penalty = (
            self.angular_divergence_penalty_weight * torso_forward_divergence
        )

        angular_speed = float(np.linalg.norm(root_ang_vel))

        ## potential based shaping
        velocity_potential = -non_forward_speed
        if self._prev_velocity_potential is None:
            velocity_shaping = 0.0
        else:
            velocity_shaping = self.velocity_shaping_weight * (
                self.velocity_shaping_gamma * velocity_potential
                - self._prev_velocity_potential
            )
        self._prev_velocity_potential = velocity_potential

        angular_velocity_potential = -angular_speed
        if self._prev_angular_velocity_potential is None:
            angular_velocity_shaping = 0.0
        else:
            angular_velocity_shaping = self.angular_velocity_shaping_weight * (
                self.angular_velocity_shaping_gamma * angular_velocity_potential
                - self._prev_angular_velocity_potential
            )
        self._prev_angular_velocity_potential = angular_velocity_potential

        graph_energy_penalty = self.graph_energy_penalty_weight * float(
            np.mean(action**2)
        )


        safe_window_reward, com_inside_window, com_window_outside_distance = (
            self._com_safe_window_reward()
        )
        (
            upper_body_above_end_effectors_reward,
            upper_body_end_effector_clearance,
        ) = self._upper_body_above_end_effectors_reward()

        ##-------- Reward -------##
        reward = (
            com_alignment_reward
            + upright_reward
            - velocity_penalty
            - angular_velocity_penalty
            - terminal_penalty
            - energy_penalty
        )

        reward += (
            -angular_divergence_penalty
            + velocity_shaping
            + angular_velocity_shaping
            + safe_window_reward
            + upper_body_above_end_effectors_reward
            - graph_energy_penalty
        )


        info["alive_reward"] = 0.0
        info["velocity_penalty"] = float(velocity_penalty)
        info["upright_reward"] = float(upright_reward)
        info["morph_params"] = self.morph
        info["velocity_shaping"] = float(velocity_shaping)
        info["angular_velocity_shaping"] = float(angular_velocity_shaping)
        info["non_forward_speed"] = float(non_forward_speed)
        info["angular_speed"] = float(angular_speed)
        info["com_reward"] = float(safe_window_reward)
        info["com_inside_limb_window"] = bool(com_inside_window)
        info["com_window_outside_distance"] = float(com_window_outside_distance)
        info["torso_forward_divergence"] = float(torso_forward_divergence)
        info["angular_penalty"] = float(angular_divergence_penalty)
        info["energy_penalty"] = float(energy_penalty)
        info["upper_body_above_reward"] = float( upper_body_above_end_effectors_reward)
        info["upper_body_clearance"] = float(  upper_body_end_effector_clearance)



        return self._flat_to_graph_obs(obs), reward, terminated, truncated, info

    def _get_obs(self):
        obs = super()._get_obs()
        phase = (self._phase_step % self.phase_cycle) / self.phase_cycle
        return np.concatenate([obs, np.array([phase], dtype=obs.dtype)])
