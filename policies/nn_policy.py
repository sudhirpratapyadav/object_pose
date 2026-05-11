"""Neural network policy: loading and inference for sim2real deployment.

Classes
-------
EmpiricalNormalization  — running-mean/std obs normalizer (matches rsl_rl)
DeployPolicy            — standalone actor MLP, loaded from a PPO checkpoint
PolicyAgent             — high-level wrapper: load → infer → return raw actions

The policy outputs raw action values. Any frame transformation (e.g. adding a
home-pose offset) is the caller's responsibility and lives in the environment /
sim2real script, not here.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict


class EmpiricalNormalization(nn.Module):
    """Running-mean / running-std observation normalizer (matches rsl_rl)."""

    def __init__(self, shape: int, eps: float = 1e-2):
        super().__init__()
        self.eps = eps
        self.register_buffer("_mean",  torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var",   torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std",   torch.ones(shape).unsqueeze(0))
        self.register_buffer("count",  torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std + self.eps)


class DeployPolicy(nn.Module):
    """Standalone actor MLP for deployment.

    Accepts a TensorDict keyed by ``obs_key`` and returns deterministic actions.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
        obs_key: str = "actor",
    ):
        super().__init__()
        self.obs_key = obs_key
        self.obs_normalizer = EmpiricalNormalization(obs_dim)

        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ELU())
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.mlp = nn.Sequential(*layers)

    @torch.no_grad()
    def forward(self, obs: TensorDict) -> torch.Tensor:
        x = obs[self.obs_key]
        x = self.obs_normalizer(x)
        return self.mlp(x)

    @staticmethod
    def from_checkpoint(
        path: str,
        device: str = "cpu",
        obs_key: str = "actor",
    ) -> "DeployPolicy":
        """Load a trained rsl_rl PPO checkpoint."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        actor_sd = ckpt["actor_state_dict"]

        obs_dim = actor_sd["mlp.0.weight"].shape[1]
        linear_keys = sorted(
            [k for k in actor_sd if k.startswith("mlp.") and k.endswith(".weight")],
            key=lambda k: int(k.split(".")[1]),
        )
        action_dim  = actor_sd[linear_keys[-1]].shape[0]
        hidden_dims = tuple(actor_sd[k].shape[0] for k in linear_keys[:-1])

        policy = DeployPolicy(obs_dim, action_dim, hidden_dims, obs_key)
        filtered = {k: v for k, v in actor_sd.items()
                    if k != "std" and not k.startswith("distribution.")}
        policy.load_state_dict(filtered, strict=True)
        policy.to(device).eval()
        return policy


class PolicyAgent:
    """High-level agent: load policy checkpoint and run inference.

    Returns the raw action vector from the network. Any coordinate-frame
    transformation (e.g. adding a home-pose offset to convert actions into
    world-frame EE targets) must be done by the caller.

    Args:
        checkpoint_path: Path to rsl_rl PPO checkpoint (.pt file).
        device:          Torch device string, e.g. ``"cpu"`` or ``"cuda:0"``.
        obs_key:         TensorDict key expected by the policy (default ``"actor"``).

    Usage::

        agent = PolicyAgent(path, device="cuda:0")
        obs    = np.random.randn(agent.obs_dim).astype(np.float32)
        action = agent.get_action(obs)   # (action_dim,) raw numpy array
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        obs_key: str = "actor",
    ):
        self.policy = DeployPolicy.from_checkpoint(
            checkpoint_path, device=device, obs_key=obs_key
        )
        self.obs_dim    = self.policy.obs_normalizer._mean.shape[1]
        self.action_dim = self.policy.mlp[-1].out_features
        self._device    = device
        self._obs_key   = obs_key

        print(
            f"[PolicyAgent] loaded  obs_dim={self.obs_dim}  "
            f"action_dim={self.action_dim}  device={device}"
        )

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Run one policy inference step.

        Args:
            obs: ``(obs_dim,)`` single observation  OR
                 ``(B, obs_dim)`` batch of observations.

        Returns:
            action: ``(action_dim,)`` for single input, or ``(B, action_dim)`` for batch.
        """
        single = obs.ndim == 1
        obs_np = obs.astype(np.float32)
        if single:
            obs_np = obs_np[None]          # (1, obs_dim)
        obs_t  = torch.from_numpy(obs_np).to(self._device)   # (B, obs_dim)
        B      = obs_t.shape[0]
        obs_td = TensorDict({self._obs_key: obs_t}, batch_size=[B])
        out    = self.policy(obs_td).cpu().numpy()            # (B, action_dim)
        return out[0] if single else out
