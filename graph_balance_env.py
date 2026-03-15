import numpy as np
from collections import deque
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
        com_safe_window_radius=0.2,
        energy_penalty_weight=0.02,
        angular_velocity_penalty_weight=0.08,
        com_alignment_weight=4,
        torso_position_stability_reward_weight=4,
        torso_position_stability_buffer=0.05,
        com_progress_weight=1.5,
        angular_divergence_penalty_weight=1.0,
        torso_height_contact_reward_weight=1.5,
        downward_velocity_shaping_weight=2,
        min_tilt_failure_height_ratio=0.4,
        min_tilt_failure_height_floor=0.4,
        unhealthy_torso_height_ratio=0.25,
        alive_weight=1,
        **kwargs,
    ):
        self.node_feature_dim = int(node_feature_dim)
        self.global_feature_dim = int(global_feature_dim)
        self.reset_height_step = float(reset_height_step)
        self.reset_max_drop = float(reset_max_drop)
        self.com_safe_window_weight = float(com_safe_window_weight)
        self.com_safe_window_progress_weight = float(com_safe_window_progress_weight)
        self.com_safe_window_radius = float(com_safe_window_radius)
        self.graph_energy_penalty_weight = float(energy_penalty_weight)
        self.energy_penalty_weight = float(energy_penalty_weight)
        self.angular_velocity_penalty_weight = float(angular_velocity_penalty_weight)
        self.com_alignment_weight = float(com_alignment_weight)
        self.torso_position_stability_reward_weight = float(
            torso_position_stability_reward_weight
        )
        self.torso_position_stability_buffer = float(torso_position_stability_buffer)
        self._torso_position_window = deque(maxlen=5)
        self.com_progress_weight = float(com_progress_weight)
        self.angular_divergence_penalty_weight = float(
            angular_divergence_penalty_weight
        )
        self.torso_height_contact_reward_weight = float(
            torso_height_contact_reward_weight
        )
        self.downward_velocity_shaping_weight = float(
            downward_velocity_shaping_weight
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
        self._prev_downward_velocity = None
        self._steps_alive = 0
        self._phase_step = 0
        self.phase_cycle = 200
        self.morph = morph_params
        self.alive_weight = alive_weight

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
        self._internal_body_ids = self._find_internal_body_ids()
        self._limb_end_viz_body_ids, self._limb_end_geom_ids = self._find_limb_end_geom_ids()
        self._leaf_node_rgba = np.array([0.0, 0.1, 1.0, 1.0], dtype=float)
        self._torso_safe_rgba = np.array([0.0, 1.0, 0.0, 1.0], dtype=float)
        self._torso_alert_rgba = np.array([1.0, 0.0, 0.0, 1.0], dtype=float)
        self._torso_geom_id = self._find_torso_geom_id()
        self._ground_geom_ids = self._find_ground_geom_ids()


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

    def _find_internal_body_ids(self):
        """Return non-leaf body ids, excluding world body."""
        child_counts = np.zeros(self.model.nbody, dtype=np.int32)
        for body_id in range(1, self.model.nbody):
            parent_id = int(self.model.body_parentid[body_id])
            if parent_id >= 0:
                child_counts[parent_id] += 1

        internal_body_ids = [
            body_id
            for body_id in range(1, self.model.nbody)
            if child_counts[body_id] > 0
        ]
        return tuple(internal_body_ids)

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

    def _find_internal_geom_ids(self):
            """Return geom ids associated with each limb-end body, in matching order."""
            body_ids = []
            geom_ids = []
            for body_id in self._internal_body_ids:
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


    def _has_unhealthy_ground_contact(self):

        ground_geom_ids = set(self._ground_geom_ids)
        #non end effectors are unsafe
        body_geom_ids = self._find_internal_geom_ids()

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

    def _has_end_effector_ground_contact(self):
        if not self._ground_geom_ids or not self._limb_end_geom_ids:
            return False

        ground_geom_ids = set(self._ground_geom_ids)
        end_effector_geom_ids = set(self._limb_end_geom_ids)

        for contact_idx in range(self.data.ncon):
            contact = self.data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)

            if (
                (geom1 in end_effector_geom_ids and geom2 in ground_geom_ids)
                or (geom2 in end_effector_geom_ids and geom1 in ground_geom_ids)
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

    def _update_com_component_visualization(self, reward_fraction):
        if not self._limb_end_geom_ids:
            return

        blend_strength = float(np.clip(reward_fraction, 0.0, 1.0))
        torso_color = (
            (1.0 - blend_strength) * self._torso_alert_rgba
            + blend_strength * self._torso_safe_rgba
        )

        for geom_id in self._limb_end_geom_ids:
            self._set_limb_end_geom_rgba(geom_id, self._leaf_node_rgba)
        self._set_torso_geom_rgba(torso_color)

    def _compute_weighted_com_window_center(self):
        if not self._limb_end_body_ids:
            return np.asarray(self.data.subtree_com[0][:2], dtype=float)

        limb_end_positions = self.data.xipos[list(self._limb_end_body_ids), :3]
        limb_end_xy = limb_end_positions[:, :2]
        limb_end_height = limb_end_positions[:, 2]

        if limb_end_height.size == 0:
            return np.asarray(self.data.subtree_com[0][:2], dtype=float)

        height_span = float(np.ptp(limb_end_height))
        if height_span <= 1e-6:
            weights = np.ones_like(limb_end_height, dtype=float)
        else:
            # Lower end effectors have smaller z values and should carry more weight.
            normalized_height = (limb_end_height - float(limb_end_height.min())) / height_span
            weights = 1.0 - normalized_height
            weights += 1e-3

        return np.average(limb_end_xy, axis=0, weights=weights)

    def _com_safe_window_reward(self):
        """Reward CoM when it stays inside a fixed-radius circle around weighted end-effectors."""
        if not self._limb_end_body_ids:
            return 0.0, False, 0.0

        com_xy = np.asarray(self.data.subtree_com[0][:2], dtype=float)
        window_center = self._compute_weighted_com_window_center()
        window_radius = max(self.com_safe_window_radius, 1e-6)
        com_window_distance = float(np.linalg.norm(com_xy - window_center))
        inside_window = com_window_distance <= window_radius
        outside_distance = max(0.0, com_window_distance - window_radius)

        # Linear falloff from center of safe window; reaches 0 at window boundary.
        center_alignment = float(
            np.clip(1.0 - (com_window_distance / window_radius), 0.0, 1.0)
        )
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
        self._update_com_component_visualization(reward_fraction)

        return safe_window_reward, inside_window, outside_distance


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
        self._prev_downward_velocity = None
        self._phase_step = 0
        self._lower_to_ground_contact_threshold()
        self._set_morphology_aware_healthy_z_range()
        torso_body_id = self.model.body("torso").id
        self._starting_torso_height = float(self.data.xipos[torso_body_id][2])
        torso_position = np.asarray(self.data.xipos[torso_body_id], dtype=float).copy()
        self._torso_position_window.clear()
        self._torso_position_window.append(torso_position)
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
        alive_reward = 0.0

        # Balance reward terms (without alive reward).

        survival_frac = np.clip(self._steps_alive / 1000, 0.0, 1.0)
        max_penalty = 100.0
        terminal_penalty = max_penalty * (1.0 - survival_frac)
        if not terminated:  # only if it actually fell, not time-limit
            terminal_penalty = 0.0

        root_lin_vel = np.asarray(self.data.qvel[0:3], dtype=float)
        downward_velocity = max(0.0, float(-root_lin_vel[2]))
        if self._prev_downward_velocity is None:
            downward_velocity_shaping = 0.0
        else:
            downward_velocity_shaping = (self.downward_velocity_shaping_weight * max(0,(self._prev_downward_velocity - downward_velocity)))
        self._prev_downward_velocity = downward_velocity

        root_ang_vel = np.asarray(self.data.qvel[3:6], dtype=float)
        angular_velocity_penalty = (
            self.angular_velocity_penalty_weight * np.linalg.norm(root_ang_vel)
        )

        action = np.asarray(action, dtype=float)
        torso_body_id = self.model.body("torso").id
        torso_position = np.asarray(self.data.xipos[torso_body_id], dtype=float)
        if self._torso_position_window:
            torso_position_avg = np.mean(self._torso_position_window, axis=0)
            torso_position_deviation = float(
                np.linalg.norm(torso_position - torso_position_avg)
            )
        else:
            torso_position_deviation = 0.0

        torso_position_excess = max(
            0.0, torso_position_deviation - self.torso_position_stability_buffer
        )
        torso_position_stability_reward = (
            self.torso_position_stability_reward_weight
            * np.exp(-torso_position_excess)
        )
        self._torso_position_window.append(torso_position.copy())

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

        graph_energy_penalty = self.graph_energy_penalty_weight * float(
            np.mean(action**2)
        )

        safe_window_reward, com_inside_window, com_window_outside_distance = (
            self._com_safe_window_reward()
        )

        end_effector_ground_contact = self._has_end_effector_ground_contact()
        if not (terminated or truncated) and end_effector_ground_contact:
            alive_reward = self.alive_weight

        torso_height_reward = 0.0
        torso_height = float(torso_position[2])
        if (
            not (terminated or truncated)
            and end_effector_ground_contact
            and self._starting_torso_height is not None
            and self._starting_torso_height > 0.0
        ):
            torso_height_ratio = torso_height / self._starting_torso_height
            torso_height_reward = (
                self.torso_height_contact_reward_weight * torso_height_ratio
            )

        ##-------- Reward -------##
        reward = (
            torso_position_stability_reward
            + alive_reward
            + torso_height_reward
            + safe_window_reward
            + downward_velocity_shaping
            - angular_velocity_penalty
            - angular_divergence_penalty
            - graph_energy_penalty
            - terminal_penalty
        )

        info["alive_reward"] = float(alive_reward)
        info["torso_position_stability_reward"] = float(torso_position_stability_reward)
        info["torso_position_deviation"] = float(torso_position_deviation)
        info["torso_position_stability_buffer"] = float(self.torso_position_stability_buffer)
        info["morph_params"] = self.morph
        info["angular_speed"] = float(angular_speed)
        info["com_reward"] = float(safe_window_reward)
        info["com_inside_limb_window"] = bool(com_inside_window)
        info["com_window_outside_distance"] = float(com_window_outside_distance)
        info["torso_forward_divergence"] = float(torso_forward_divergence)
        info["angular_penalty"] = float(angular_divergence_penalty)
        info["end_effector_ground_contact"] = bool(end_effector_ground_contact)
        info["torso_height"] = float(torso_height)
        info["torso_height_reward"] = float(torso_height_reward)
        info["vertical_velocity_shaping"] = float(downward_velocity_shaping)


        return self._flat_to_graph_obs(obs), reward, terminated, truncated, info

    def _get_obs(self):
        obs = super()._get_obs()
        phase = (self._phase_step % self.phase_cycle) / self.phase_cycle
        return np.concatenate([obs, np.array([phase], dtype=obs.dtype)])
