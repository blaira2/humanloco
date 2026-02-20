import numpy as np
from gymnasium import spaces

from balance_humanoid_env import BalanceHumanoidEnv


class GraphBalanceHumanoidEnv(BalanceHumanoidEnv):
    """BalanceHumanoidEnv that emits graph-structured observations for GCN policies."""

    def __init__(
        self,
        xml_file=None,
        morph_params=None,
        node_feature_dim=8,
        global_feature_dim=32,
        reset_height_step=0.0025,
        reset_max_drop=0.2,
        com_safe_window_weight=1,
        com_safe_window_progress_weight=0.5,
        velocity_shaping_weight=0.5,
        velocity_shaping_gamma=0.99,
        angular_velocity_shaping_weight=0.5,
        angular_velocity_shaping_gamma=0.99,
        min_tilt_failure_height_ratio=0.55,
        min_tilt_failure_height_floor=0.6,
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
        self.angular_velocity_shaping_weight = float(angular_velocity_shaping_weight)
        self.angular_velocity_shaping_gamma = float(angular_velocity_shaping_gamma)
        self._min_tilt_failure_height_ratio = float(min_tilt_failure_height_ratio)
        self._min_tilt_failure_height_floor = float(min_tilt_failure_height_floor)
        self._use_morphology_aware_healthy_z_range = "healthy_z_range" not in kwargs
        self._prev_velocity_potential = None
        self._prev_angular_velocity_potential = None
        self._prev_com_window_distance = None

        super().__init__(xml_file=xml_file, morph_params=morph_params, **kwargs)

        self._num_nodes = int(self.model.nbody - 1)  # skip world body
        self._adjacency = self._build_adjacency().astype(np.float32)
        self._limb_end_body_ids = self._find_limb_end_body_ids()

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

        if inside_window:
            safe_window_reward = self.com_safe_window_weight
        else:
            safe_window_reward = 0.0

        if self._prev_com_window_distance is None:
            progress_reward = 0.0
        else:
            progress_reward = self.com_safe_window_progress_weight * (
                self._prev_com_window_distance - com_window_distance
            )
        self._prev_com_window_distance = com_window_distance

        safe_window_reward += progress_reward

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
        self._lower_to_ground_contact_threshold()
        self._set_morphology_aware_healthy_z_range()
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
        obs, reward, terminated, truncated, info = super().step(action)

        root_lin_vel = np.asarray(self.data.qvel[0:3], dtype=float)
        non_forward_components = np.array(
            [max(0.0, -root_lin_vel[0]), root_lin_vel[1], root_lin_vel[2]],
            dtype=float,
        )
        non_forward_speed = float(np.linalg.norm(non_forward_components))

        root_ang_vel = np.asarray(self.data.qvel[3:6], dtype=float)
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


        safe_window_reward, com_inside_window, com_window_outside_distance = (
            self._com_safe_window_reward()
        )

        ##-------- Reward -------##
        reward += (velocity_shaping
                   + angular_velocity_shaping
                   + safe_window_reward)


        info["velocity_shaping"] = float(velocity_shaping)
        info["angular_velocity_shaping"] = float(angular_velocity_shaping)
        info["non_forward_speed"] = float(non_forward_speed)
        info["angular_speed"] = float(angular_speed)
        info["com_safe_window_reward"] = float(safe_window_reward)
        info["com_inside_limb_window"] = bool(com_inside_window)
        info["com_window_outside_distance"] = float(com_window_outside_distance)

        return self._flat_to_graph_obs(obs), reward, terminated, truncated, info
