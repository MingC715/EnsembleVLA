"""Energy head used by the released EnsembleVLA checkpoints."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableCompositionWeights(nn.Module):
    def __init__(self, init_w1: float = 0.3, init_w2: float = 0.7):
        super().__init__()
        logits = torch.tensor([np.log(init_w1 + 1e-8), np.log(init_w2 + 1e-8)], dtype=torch.float32)
        self.logits = nn.Parameter(logits)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = F.softmax(self.logits, dim=0)
        return weights[0], weights[1]

    def get_weights(self) -> Tuple[float, float]:
        with torch.no_grad():
            weights = F.softmax(self.logits, dim=0)
            return weights[0].item(), weights[1].item()

    def __repr__(self) -> str:
        w1, w2 = self.get_weights()
        return f"LearnableCompositionWeights(w1={w1:.4f}, w2={w2:.4f})"


class ConservativeResidualHead(nn.Module):
    def __init__(
        self,
        action_dim: int = 14,
        hidden_dim: int = 64,
        num_layers: int = 2,
        max_delta: float = 0.001,
        use_spectral_norm: bool = True,
        init_gate_bias: float = -2.0,
        use_base_context: bool = False,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.max_delta = max_delta
        self.init_gate_bias = init_gate_bias
        self.use_base_context = use_base_context
        in_dim = action_dim * 3 if use_base_context else action_dim

        delta_layers = []
        cur_dim = in_dim
        for layer_idx in range(num_layers):
            out_dim = hidden_dim if layer_idx < num_layers - 1 else action_dim
            linear = nn.Linear(cur_dim, out_dim)
            if use_spectral_norm and layer_idx < num_layers - 1:
                linear = nn.utils.spectral_norm(linear)
            delta_layers.append(linear)
            if layer_idx < num_layers - 1:
                delta_layers.append(nn.SiLU())
            cur_dim = hidden_dim
        delta_layers.append(nn.Tanh())
        self.delta_net = nn.Sequential(*delta_layers)

        self.gate_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        with torch.no_grad():
            self.gate_net[-1].bias.fill_(init_gate_bias)

    def _features(
        self,
        composed_action: torch.Tensor,
        base1: Optional[torch.Tensor] = None,
        base2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.use_base_context:
            return composed_action
        if base1 is None:
            base1 = composed_action
        if base2 is None:
            base2 = composed_action
        return torch.cat([composed_action, base1, base2], dim=-1)

    def forward(
        self,
        composed_action: torch.Tensor,
        base1: Optional[torch.Tensor] = None,
        base2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self._features(composed_action, base1, base2)
        delta = self.delta_net(feat) * self.max_delta
        gate = torch.sigmoid(self.gate_net(feat))
        return delta, gate

    def refine_action(
        self,
        composed_action: torch.Tensor,
        base1: Optional[torch.Tensor] = None,
        base2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if composed_action.dim() == 3:
            batch, horizon, dim = composed_action.shape
            flat = composed_action.reshape(batch * horizon, dim)
            b1 = base1.reshape(batch * horizon, dim) if base1 is not None else None
            b2 = base2.reshape(batch * horizon, dim) if base2 is not None else None
            delta, gate = self(flat, b1, b2)
            return (flat + gate * delta).reshape(batch, horizon, dim)
        delta, gate = self(composed_action, base1, base2)
        return composed_action + gate * delta

    def get_gate_stats(self) -> dict:
        with torch.no_grad():
            bias = self.gate_net[-1].bias.item()
            return {"gate_bias": bias, "estimated_init_gate": torch.sigmoid(torch.tensor(bias)).item()}


class ConservativeEnergyHead(nn.Module):
    def __init__(
        self,
        action_dim: int = 14,
        hidden_dim: int = 64,
        num_layers: int = 2,
        max_delta: float = 0.001,
        use_spectral_norm: bool = True,
        init_w1: float = 0.3,
        init_w2: float = 0.7,
        init_gate_bias: float = -2.0,
        use_base_context: bool = False,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.max_delta = max_delta
        self.use_base_context = use_base_context
        self.composition_weights = LearnableCompositionWeights(init_w1=init_w1, init_w2=init_w2)
        self.residual_head = ConservativeResidualHead(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_delta=max_delta,
            use_spectral_norm=use_spectral_norm,
            init_gate_bias=init_gate_bias,
            use_base_context=use_base_context,
        )

    def get_composition_weights(self):
        return self.composition_weights()

    def get_weights_for_display(self):
        return self.composition_weights.get_weights()

    def forward(
        self,
        composed_action: torch.Tensor,
        base1: Optional[torch.Tensor] = None,
        base2: Optional[torch.Tensor] = None,
    ):
        return self.residual_head(composed_action, base1, base2)

    def refine_action(
        self,
        composed_action: torch.Tensor,
        base1: Optional[torch.Tensor] = None,
        base2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.residual_head.refine_action(composed_action, base1, base2)

    def compute_loss(
        self,
        expert_action: torch.Tensor,
        composed_action: torch.Tensor,
        base1: Optional[torch.Tensor] = None,
        base2: Optional[torch.Tensor] = None,
    ) -> tuple:
        delta, gate = self(composed_action, base1, base2)
        refined = composed_action + gate * delta
        mse_loss = F.mse_loss(refined, expert_action)

        with torch.no_grad():
            mse_composed = ((composed_action - expert_action) ** 2).mean(dim=-1)
            mse_refined = ((refined.detach() - expert_action) ** 2).mean(dim=-1)
            improvement = mse_composed - mse_refined

        gate_target = (improvement > 0).float().unsqueeze(-1)
        gate_loss = F.binary_cross_entropy(gate, gate_target)
        conservative_penalty = F.relu(-improvement).mean()
        target_delta = torch.clamp(expert_action - composed_action, -self.max_delta, self.max_delta)
        delta_supervision_loss = F.mse_loss(delta, target_delta)
        delta_reg = 0.001 * (delta ** 2).mean()

        total_loss = mse_loss + 0.1 * gate_loss + 0.5 * delta_supervision_loss + 0.1 * conservative_penalty + delta_reg
        with torch.no_grad():
            w1, w2 = self.get_weights_for_display()
            metrics = {
                "mse_loss": mse_loss.item(),
                "gate_loss": gate_loss.item(),
                "delta_supervision_loss": delta_supervision_loss.item(),
                "conservative_penalty": conservative_penalty.item(),
                "delta_reg": delta_reg.item(),
                "mse_composed": mse_composed.mean().item(),
                "mse_refined": mse_refined.mean().item(),
                "improvement": improvement.mean().item(),
                "improvement_rate": (improvement > 0).float().mean().item(),
                "gate_mean": gate.mean().item(),
                "gate_std": gate.std().item(),
                "delta_norm": delta.norm(dim=-1).mean().item(),
                "w1": w1,
                "w2": w2,
                **self.residual_head.get_gate_stats(),
            }
        return total_loss, metrics

    def print_weights(self, prefix: str = "") -> None:
        w1, w2 = self.get_weights_for_display()
        gate_stats = self.residual_head.get_gate_stats()
        print(
            f"{prefix}Composition: w1={w1:.4f}, w2={w2:.4f} | "
            f"Gate bias={gate_stats['gate_bias']:.4f}, "
            f"Est. init gate={gate_stats['estimated_init_gate']:.4f}"
        )
