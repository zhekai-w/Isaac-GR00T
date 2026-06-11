# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SmolVLA-style flow-matching action head for GR00T N1.5.

This is a self-contained ("vendored") port of SmolVLA's action expert
(``lerobot/src/lerobot/policies/smolvla/``) re-targeted to GR00T's *decoupled*
backbone interface. Unlike upstream SmolVLA -- whose expert is fused with the VLM and
cross-attends the VLM's per-layer KV cache -- this head consumes GR00T's single final
feature tensor ``backbone_features (B, S, backbone_embedding_dim)`` as cross-attention
memory, and otherwise honors the same contract as ``FlowmatchingActionHead`` so it can be
swapped in behind ``GR00T_N1_5`` without touching the backbone or data pipeline.

Design decisions (single-robot SmolVLA flavor):
  * Single ``nn.Linear`` projections (no per-embodiment ``CategorySpecificMLP``);
    ``embodiment_id`` is ignored.
  * State enters the expert as a prepended token (the backbone has already run, so it
    cannot go into the VLM prefix as upstream SmolVLA does).
  * Flow-matching with SmolVLA conventions: ``x_t = t*noise + (1-t)*action``,
    target velocity ``u_t = noise - action``, ``Beta(1.5, 1.0)`` time sampling,
    ``num_steps``-step Euler with negative dt (t: 1 -> 0).
  * Expert block is RMSNorm + RoPE causal self-attn + cross-attn to backbone memory +
    SwiGLU MLP, alternating self / cross attention by ``self_attn_every_n_layers``.
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature


# --------------------------------------------------------------------------------------
# Helpers (ported from lerobot smolvla)
# --------------------------------------------------------------------------------------
def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> torch.Tensor:
    """Sine-cosine positional embedding for scalar positions. ``time`` is shape (B,)."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None].float()
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def get_intermediate_size(hidden_dim, ffn_dim_multiplier=4, multiple_of=256):
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, max_wavelength: float = 10_000.0):
    """Apply RoPE. ``x``: (B, L, H, D), ``positions``: (B, L)."""
    d_half = x.shape[-1] // 2
    device, dtype = x.device, x.dtype
    x = x.to(torch.float32)
    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].float() / timescale[None, None, :]
    radians = radians[..., None, :]  # (B, L, 1, d_half)
    sin, cos = torch.sin(radians), torch.cos(radians)
    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin
    return res.to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate, bias=False)
        self.up_proj = nn.Linear(dim, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ExpertBlock(nn.Module):
    """One transformer block. ``is_self_attn`` toggles causal self-attn (RoPE over expert
    tokens) vs cross-attn to backbone memory (K/V projected from ``memory_dim``)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        memory_dim: int,
        is_self_attn: bool,
        intermediate_size: int,
    ):
        super().__init__()
        self.is_self_attn = is_self_attn
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must divide num_heads"

        self.input_norm = RMSNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        kv_in = hidden_size if is_self_attn else memory_dim
        self.k_proj = nn.Linear(kv_in, hidden_size, bias=False)
        self.v_proj = nn.Linear(kv_in, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.post_attn_norm = RMSNorm(hidden_size)
        self.mlp = SwiGLU(hidden_size, intermediate_size)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, hidden) -> (B, L, num_heads, head_dim)
        b, l, _ = x.shape
        return x.view(b, l, self.num_heads, self.head_dim)

    def forward(self, hidden, memory=None, memory_mask=None, positions=None):
        residual = hidden
        h = self.input_norm(hidden)

        q = self._shape(self.q_proj(h))
        if self.is_self_attn:
            k = self._shape(self.k_proj(h))
            v = self._shape(self.v_proj(h))
            if positions is not None:
                q = apply_rope(q, positions)
                k = apply_rope(k, positions)
            attn_mask, is_causal = None, True
        else:
            k = self._shape(self.k_proj(memory))
            v = self._shape(self.v_proj(memory))
            # memory_mask: (B, S) bool, True = valid -> (B, 1, 1, S)
            attn_mask = memory_mask[:, None, None, :] if memory_mask is not None else None
            is_causal = False

        # (B, L, Hd, D) -> (B, Hd, L, D)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        out = out.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[1], -1)
        hidden = residual + self.o_proj(out)

        hidden = hidden + self.mlp(self.post_attn_norm(hidden))
        return hidden


# --------------------------------------------------------------------------------------
# Config + head
# --------------------------------------------------------------------------------------
@dataclass
class SmolVLAActionHeadConfig(PretrainedConfig):
    action_dim: int = field(default=None)
    action_horizon: int = field(default=None)
    max_state_dim: int = field(default=None)
    max_action_dim: int = field(default=None)

    backbone_embedding_dim: int = field(default=1536)
    expert_hidden_size: int = field(default=720)
    num_layers: int = field(default=16)
    num_heads: int = field(default=12)
    self_attn_every_n_layers: int = field(default=2)

    num_steps: int = field(default=10)
    min_period: float = field(default=4e-3)
    max_period: float = field(default=4.0)
    noise_beta_alpha: float = field(default=1.5)
    noise_beta_beta: float = field(default=1.0)
    time_sampling_s: float = field(default=0.999)

    use_vlln: bool = field(default=True)
    add_state_token: bool = field(default=True)
    model_dtype: str = field(default="float32")
    tune_projector: bool = field(default=True)
    tune_diffusion_model: bool = field(default=True)

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        super().__init__()


class SmolVLAActionHead(nn.Module):
    config_class = SmolVLAActionHeadConfig

    def __init__(self, config: SmolVLAActionHeadConfig):
        super().__init__()
        self.config = config
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        H = config.expert_hidden_size

        # Backbone feature preprocessing (memory for cross-attention).
        self.vlln = nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()

        # Projections (SmolVLA single-robot style).
        self.state_proj = nn.Linear(config.max_state_dim, H)
        self.action_in_proj = nn.Linear(config.action_dim, H)
        self.action_time_mlp_in = nn.Linear(2 * H, H)
        self.action_time_mlp_out = nn.Linear(H, H)
        self.action_out_proj = nn.Linear(H, config.action_dim)

        # Expert transformer (alternating self / cross attention).
        intermediate = get_intermediate_size(H)
        layers = []
        for idx in range(config.num_layers):
            is_self = (
                config.self_attn_every_n_layers > 0
                and idx % config.self_attn_every_n_layers == 0
            )
            layers.append(
                ExpertBlock(
                    hidden_size=H,
                    num_heads=config.num_heads,
                    memory_dim=config.backbone_embedding_dim,
                    is_self_attn=is_self,
                    intermediate_size=intermediate,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.final_norm = RMSNorm(H)

        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

    # ----- contract: trainability / dtype / input prep -----
    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        projector = [
            self.state_proj,
            self.action_in_proj,
            self.action_time_mlp_in,
            self.action_time_mlp_out,
            self.action_out_proj,
        ]
        if not tune_projector:
            for m in projector:
                m.requires_grad_(False)
        if not tune_diffusion_model:
            self.layers.requires_grad_(False)
            self.final_norm.requires_grad_(False)
            self.vlln.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head expert: {self.tune_diffusion_model}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        if self.training:
            if not self.tune_diffusion_model:
                self.layers.eval()
                self.final_norm.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    # ----- flow-matching utilities -----
    def sample_time(self, batch_size, device, dtype):
        beta = torch.distributions.Beta(
            concentration1=self.config.noise_beta_alpha,
            concentration0=self.config.noise_beta_beta,
        )
        s = beta.sample((batch_size,)).to(device=device, dtype=dtype)
        return s * self.config.time_sampling_s + (1.0 - self.config.time_sampling_s)

    def process_backbone_output(self, backbone_output: BatchFeature) -> torch.Tensor:
        return self.vlln(backbone_output["backbone_features"])

    def embed_suffix(self, noisy_actions: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """noisy_actions: (B, H, action_dim), time: (B,) -> (B, H, expert_hidden)."""
        action_emb = self.action_in_proj(noisy_actions)
        time_emb = create_sinusoidal_pos_embedding(
            time,
            self.config.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=noisy_actions.device,
        ).to(action_emb.dtype)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        x = torch.cat([action_emb, time_emb], dim=2)
        x = self.action_time_mlp_in(x)
        x = F.silu(x)
        return self.action_time_mlp_out(x)

    def _run_expert(self, tokens: torch.Tensor, memory: torch.Tensor, memory_mask) -> torch.Tensor:
        """tokens: (B, L, expert_hidden). Returns last-`action_horizon` action features."""
        positions = torch.arange(tokens.shape[1], device=tokens.device)[None, :].expand(
            tokens.shape[0], -1
        )
        for layer in self.layers:
            if layer.is_self_attn:
                tokens = layer(tokens, positions=positions)
            else:
                tokens = layer(tokens, memory=memory, memory_mask=memory_mask)
        tokens = self.final_norm(tokens)
        return tokens[:, -self.action_horizon :]

    def _build_tokens(self, suffix_emb: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if not self.config.add_state_token:
            return suffix_emb
        state_emb = self.state_proj(state)
        if state_emb.ndim == 2:
            state_emb = state_emb[:, None, :]
        return torch.cat([state_emb, suffix_emb], dim=1)

    # ----- training -----
    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        memory = self.process_backbone_output(backbone_output)
        memory_mask = backbone_output.get("backbone_attention_mask", None)
        if memory_mask is not None:
            memory_mask = memory_mask.bool()

        actions = action_input.action
        state = action_input.state
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        time = self.sample_time(actions.shape[0], actions.device, actions.dtype)

        t = time[:, None, None]
        x_t = t * noise + (1 - t) * actions
        u_t = noise - actions  # target velocity

        suffix_emb = self.embed_suffix(x_t, time)
        tokens = self._build_tokens(suffix_emb, state)
        action_feats = self._run_expert(tokens, memory, memory_mask)
        v_t = self.action_out_proj(action_feats)

        action_mask = action_input.action_mask
        loss = F.mse_loss(v_t, u_t, reduction="none") * action_mask
        loss = loss.sum() / action_mask.sum().clamp_min(1.0)
        return BatchFeature(data={"loss": loss})

    # ----- inference -----
    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        memory = self.process_backbone_output(backbone_output)
        memory_mask = backbone_output.get("backbone_attention_mask", None)
        if memory_mask is not None:
            memory_mask = memory_mask.bool()

        state = action_input.state
        bsize = memory.shape[0]
        device, dtype = memory.device, memory.dtype
        x_t = torch.randn(
            (bsize, self.action_horizon, self.action_dim), device=device, dtype=dtype
        )

        dt = -1.0 / self.config.num_steps
        time = 1.0
        for _ in range(self.config.num_steps):
            time_tensor = torch.full((bsize,), time, device=device, dtype=dtype)
            suffix_emb = self.embed_suffix(x_t, time_tensor)
            tokens = self._build_tokens(suffix_emb, state)
            action_feats = self._run_expert(tokens, memory, memory_mask)
            v_t = self.action_out_proj(action_feats)
            x_t = x_t + dt * v_t
            time += dt
        return BatchFeature(data={"action_pred": x_t})
