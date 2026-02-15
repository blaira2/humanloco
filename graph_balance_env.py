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
        **kwargs,
    ):
        self.node_feature_dim = int(node_feature_dim)
        self.global_feature_dim = int(global_feature_dim)
        self.reset_height_step = float(reset_height_step)
        self.reset_max_drop = float(reset_max_drop)
        super().__init__(xml_file=xml_file, morph_params=morph_params, **kwargs)

        self._num_nodes = int(self.model.nbody - 1)  # skip world body
        self._adjacency = self._build_adjacency().astype(np.float32)

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

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._lower_to_ground_contact_threshold()
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
        return self._flat_to_graph_obs(obs), reward, terminated, truncated, info
