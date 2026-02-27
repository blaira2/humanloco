# GCN integration points for humanoid RL

This repo now includes a concrete graph training path for balance control.

## What was added

- `graph_balance_env.py` with `GraphBalanceHumanoidEnv`.
  - Wraps `BalanceHumanoidEnv` reward/termination behavior.
  - Converts flat MuJoCo observation into a graph dict:
    - `node_features`: `[num_nodes, node_feature_dim]`
    - `adjacency`: `[num_nodes, num_nodes]` (body-tree edges + self loops)
    - `global_features`: fixed-length global vector
- `gcn_extractor.py` with:
  - `GraphConvLayer` (normalized adjacency message passing)
  - `HumanoidGCNExtractor` (GCN stack + global MLP + readout)
- `humanoid_module.py` additions:
  - `make_graph_balance_env(...)`
  - `graph_policy_kwargs`/`graph_sac_policy_kwargs` using `HumanoidGCNExtractor`
  - `train_graph_balance_env(...)` using SAC `MultiInputPolicy`

## How to train

Use `train_graph_balance_env(...)` (instead of `train_balance_env(...)`) to train a graph policy on the balance task.

## Purpose of the GCN extractor

The GCN extractor is the feature encoder between graph observations and SAC heads.
It maps node/edge structure + global features into a dense latent vector used by actor and critic networks.

## Reward note for graph balance runs

`GraphBalanceHumanoidEnv` inherits `BalanceHumanoidEnv.step(...)`, which sets `info["alive_reward"]` to `1.0` on non-terminal steps and `0.0` on terminal/truncated steps.
So in callback logs, `alive` will often print near `1.000` for long episodes (for example, a 1000-step time-limit episode averages to ~0.999 and rounds to `1.000`).

