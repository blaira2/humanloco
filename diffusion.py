# diffusion.py

from typing import Optional
import torch
import torch.nn as nn

class PoseDiffusion(nn.Module):
    def __init__(
        self,
        D_target: int,
        D_cond: int,
        n_steps: int = 100,
        hidden: int = 512,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()
        self.D_target = D_target
        self.D_cond = D_cond
        self.n_steps = n_steps

        betas = torch.linspace(beta_start, beta_end, n_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

        self.t_embed = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

        self.net = nn.Sequential(
            nn.Linear(D_target + D_cond + hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, D_target),
        )

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """
        x_t:   (B, D_target)
        t:     (B,) int64
        cond:  (B, D_cond)
        """
        t_norm = t.float().unsqueeze(-1) / float(self.n_steps - 1)
        t_feat = self.t_embed(t_norm)
        h = torch.cat([x_t, cond, t_feat], dim=-1)
        eps_pred = self.net(h)
        return eps_pred

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_0)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1)
        return torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1.0 - alpha_bar_t) * noise

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, n_steps: Optional[int] = None) -> torch.Tensor:
        """
        Given cond (B, D_cond), sample x_0 ~ p_theta(x|cond).
        Returns (B, D_target).
        """
        if n_steps is None:
            n_steps = self.n_steps
        device = cond.device
        B = cond.shape[0]

        x_t = torch.randn(B, self.D_target, device=device)

        for t_step in reversed(range(n_steps)):
            t = torch.full((B,), t_step, device=device, dtype=torch.long)
            eps_theta = self(x_t, t, cond)

            alpha_t = self.alphas[t].view(-1, 1)
            alpha_bar_t = self.alpha_bars[t].view(-1, 1)
            beta_t = self.betas[t].view(-1, 1)

            if t_step > 0:
                z = torch.randn_like(x_t)
            else:
                z = torch.zeros_like(x_t)

            x_t = (1.0 / torch.sqrt(alpha_t)) * (
                x_t - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * eps_theta
            ) + torch.sqrt(beta_t) * z

        return x_t
