import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GraphConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, node_features, adjacency):
        # adjacency: [B, N, N], node_features: [B, N, F]
        degree = adjacency.sum(dim=-1)
        degree_inv_sqrt = torch.pow(degree.clamp(min=1e-6), -0.5)
        degree_inv_sqrt = torch.diag_embed(degree_inv_sqrt)
        norm_adj = degree_inv_sqrt @ adjacency @ degree_inv_sqrt
        agg = norm_adj @ node_features
        return self.linear(agg)


class HumanoidGCNExtractor(BaseFeaturesExtractor):
    """GCN feature extractor for dict observations from GraphBalanceHumanoidEnv."""

    def __init__(
        self,
        observation_space,
        gcn_hidden_dim=64,
        gcn_layers=2,
        global_hidden_dim=64,
        features_dim=128,
    ):
        super().__init__(observation_space, features_dim)

        node_shape = observation_space["node_features"].shape
        global_dim = observation_space["global_features"].shape[0]
        node_in_dim = int(node_shape[-1])

        self.input_proj = nn.Linear(node_in_dim, gcn_hidden_dim)
        self.gcn_stack = nn.ModuleList(
            [GraphConvLayer(gcn_hidden_dim, gcn_hidden_dim) for _ in range(gcn_layers)]
        )

        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, global_hidden_dim),
            nn.ReLU(),
            nn.Linear(global_hidden_dim, global_hidden_dim),
            nn.ReLU(),
        )
        self.readout = nn.Sequential(
            nn.Linear(gcn_hidden_dim + global_hidden_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        node_features = observations["node_features"]
        adjacency = observations["adjacency"]
        global_features = observations["global_features"]

        x = torch.relu(self.input_proj(node_features))
        for layer in self.gcn_stack:
            x = torch.relu(layer(x, adjacency))

        graph_embedding = x.mean(dim=1)
        global_embedding = self.global_mlp(global_features)
        return self.readout(torch.cat([graph_embedding, global_embedding], dim=-1))
