from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import TafsutConfig
from .layers import Encoder, InstanceNorm, Patch, ResidualBlock


class TafsutModel(nn.Module):

    def __init__(self, cfg: TafsutConfig):
        super().__init__()
        self.cfg = cfg
        act_info = cfg.feed_forward_proj.split('-')
        self.dense_act_fn = act_info[-1]
        assert act_info[0] != 'gated', 'gated activation is not supported'
        assert cfg.input_patch_size == cfg.output_patch_size, f'input_patch_size and output_patch_size must be equal, but found {cfg.input_patch_size} and {cfg.output_patch_size}'
        assert cfg.d_model == cfg.num_heads * cfg.d_kv, f'd_model ({cfg.d_model}) must equal num_heads * d_kv ({cfg.num_heads} * {cfg.d_kv})'
        if cfg.constant_atol < 0.0:
            raise ValueError(f'constant_atol must be >= 0, got {cfg.constant_atol}')
        if cfg.constant_rtol < 0.0:
            raise ValueError(f'constant_rtol must be >= 0, got {cfg.constant_rtol}')
        if not 0.0 < cfg.constant_tail_fraction < 1.0:
            raise ValueError(f'constant_tail_fraction must be strictly between 0 and 1, got {cfg.constant_tail_fraction}')
        self.num_quantiles = len(cfg.quantiles)
        self.time_encoding_scale = int(cfg.time_encoding_scale if cfg.time_encoding_scale is not None else cfg.context_length)
        self._default_num_output_patches = math.ceil(cfg.prediction_length / cfg.output_patch_size)
        self.input_patch_embedding = ResidualBlock(in_dim=cfg.input_patch_size * 3, h_dim=cfg.d_ff, out_dim=cfg.d_model, act_fn_name=self.dense_act_fn, dropout_p=cfg.dropout_rate)
        self.patch = Patch(patch_size=cfg.input_patch_size, patch_stride=cfg.input_patch_stride)
        self.instance_norm = InstanceNorm(use_arcsinh=cfg.use_arcsinh)
        self.reg_token_id = 1
        self.vocab_size = 2 if cfg.use_reg_token else 1
        self.shared = nn.Embedding(self.vocab_size, cfg.d_model)
        self.encoder = Encoder(num_layers=cfg.num_layers, d_model=cfg.d_model, d_kv=cfg.d_kv, num_heads=cfg.num_heads, d_ff=cfg.d_ff, dropout_rate=cfg.dropout_rate, dense_act_fn=self.dense_act_fn, rope_theta=cfg.rope_theta, layer_norm_epsilon=cfg.layer_norm_epsilon, attn_implementation=cfg.attn_implementation)
        self.output_patch_embedding = ResidualBlock(in_dim=cfg.d_model, h_dim=cfg.d_ff, out_dim=self.num_quantiles * cfg.output_patch_size, act_fn_name=self.dense_act_fn, dropout_p=cfg.dropout_rate)

    @classmethod
    def from_pretrained(
        cls,
        source: str,
        device: str | torch.device | None = None,
        *,
        revision: str | None = None,
        token: str | bool | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ) -> "TafsutModel":
        """Load Tafsut from a local directory or Hugging Face model repository."""
        from .checkpoint import load_tafsut

        return load_tafsut(
            source,
            device=device,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

    def save_pretrained(
        self,
        directory: str,
        *,
        safe_serialization: bool = True,
    ) -> str:
        """Save config and inference weights in a Hugging Face-style directory."""
        from .checkpoint import save_tafsut

        return str(
            save_tafsut(
                self,
                directory,
                safe_serialization=safe_serialization,
            )
        )

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _constant_fallback_candidates(self, context: torch.Tensor, context_mask: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        if context.ndim != 2:
            raise ValueError(f'context must have shape (B, T), got {tuple(context.shape)}')
        if not torch.is_floating_point(context):
            raise TypeError(f'context must be floating point, got dtype={context.dtype}')
        if context_mask is None:
            valid_mask = torch.isfinite(context)
        else:
            if context_mask.shape != context.shape:
                raise ValueError(f'context_mask must have the same shape as context: {tuple(context_mask.shape)} != {tuple(context.shape)}')
            valid_mask = context_mask.to(device=context.device, dtype=torch.bool)
            valid_mask = valid_mask & torch.isfinite(context)
        if context.shape[-1] > self.cfg.context_length:
            context = context[..., -self.cfg.context_length:]
            valid_mask = valid_mask[..., -self.cfg.context_length:]
        batch_size = context.shape[0]
        fallback_mask = torch.zeros(batch_size, dtype=torch.bool, device=context.device)
        fallback_value = torch.full((batch_size, 1), float('nan'), dtype=torch.float32, device=context.device)
        for batch_idx in range(batch_size):
            observed = context[batch_idx][valid_mask[batch_idx]].to(torch.float32)
            num_observed = int(observed.numel())
            if num_observed == 0:
                continue
            full_level = observed.median()
            full_is_constant = bool(torch.isclose(observed, full_level, rtol=self.cfg.constant_rtol, atol=self.cfg.constant_atol).all())
            last_value = observed[-1]
            near_last = torch.isclose(observed, last_value, rtol=self.cfg.constant_rtol, atol=self.cfg.constant_atol)
            trailing_count = int(torch.cumprod(near_last.flip(0).to(torch.int64), dim=0).sum().item())
            tail_is_majority = trailing_count > self.cfg.constant_tail_fraction * num_observed
            if full_is_constant:
                fallback_mask[batch_idx] = True
                fallback_value[batch_idx, 0] = full_level
            elif tail_is_majority:
                fallback_mask[batch_idx] = True
                fallback_value[batch_idx, 0] = observed[-trailing_count:].median()
        return (fallback_mask, fallback_value)

    def _prepare_patched_context(self, context: torch.Tensor, context_mask: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if context.ndim != 2:
            raise ValueError(f'context must have shape (B, T), got {tuple(context.shape)}')
        if not torch.is_floating_point(context):
            raise TypeError(f'context must be floating point, got dtype={context.dtype}')
        if context_mask is None:
            valid_mask = torch.isfinite(context)
        else:
            if context_mask.shape != context.shape:
                raise ValueError(f'context_mask must have the same shape as context: {tuple(context_mask.shape)} != {tuple(context.shape)}')
            valid_mask = context_mask.to(device=context.device, dtype=torch.bool)
            valid_mask = valid_mask & torch.isfinite(context)
        batch_size, context_length = context.shape
        if context_length > self.cfg.context_length:
            context = context[..., -self.cfg.context_length:]
            valid_mask = valid_mask[..., -self.cfg.context_length:]
        context_for_norm = context.masked_fill(~valid_mask, float('nan'))
        context, loc_scale = self.instance_norm(context_for_norm)
        context = context.to(self.dtype)
        context_mask = valid_mask.to(self.dtype)
        patched_context = self.patch(context)
        patched_mask = torch.nan_to_num(self.patch(context_mask), nan=0.0)
        patched_context = torch.where(patched_mask > 0.0, patched_context, torch.zeros_like(patched_context))
        attention_mask = patched_mask.sum(dim=-1) > 0
        num_context_patches = attention_mask.shape[-1]
        final_context_length = num_context_patches * self.cfg.input_patch_size
        context_time_enc = torch.arange(start=-final_context_length, end=0, device=self.device, dtype=torch.float32)
        context_time_enc = context_time_enc.view(1, num_context_patches, self.cfg.input_patch_size).expand(batch_size, -1, -1).div(self.time_encoding_scale).to(self.dtype)
        patched_context = torch.cat([context_time_enc, patched_context, patched_mask], dim=-1)
        return (patched_context, attention_mask, loc_scale)

    def _prepare_patched_future(self, loc_scale: Tuple[torch.Tensor, torch.Tensor], num_output_patches: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        P = self.cfg.output_patch_size
        patched_future_covariates = torch.zeros(batch_size, num_output_patches, P, device=self.device, dtype=self.dtype)
        patched_future_covariates_mask = torch.zeros(batch_size, num_output_patches, P, device=self.device, dtype=self.dtype)
        final_future_length = num_output_patches * P
        future_time_enc = torch.arange(0, final_future_length, device=self.device, dtype=torch.float32)
        future_time_enc = future_time_enc.view(1, num_output_patches, P).expand(batch_size, -1, -1).div(self.time_encoding_scale).to(self.dtype)
        patched_future = torch.cat([future_time_enc, patched_future_covariates, patched_future_covariates_mask], dim=-1)
        return (patched_future, patched_future_covariates_mask)

    def forward(self, context: torch.Tensor, context_mask: Optional[torch.Tensor]=None, num_output_patches: Optional[int]=None, use_constant_series_fallback: Optional[bool]=None) -> torch.Tensor:
        if num_output_patches is None:
            num_output_patches = self._default_num_output_patches
        if context.ndim != 2:
            raise ValueError(f'context must be (B, T), got {tuple(context.shape)}')
        batch_size = context.shape[0]
        if use_constant_series_fallback is None:
            apply_constant_fallback = self.cfg.use_constant_series_fallback
        else:
            apply_constant_fallback = bool(use_constant_series_fallback)
        if apply_constant_fallback:
            constant_fallback_mask, constant_fallback_value = self._constant_fallback_candidates(context, context_mask)
        else:
            constant_fallback_mask = torch.zeros(batch_size, dtype=torch.bool, device=context.device)
            constant_fallback_value = torch.full((batch_size, 1), float('nan'), dtype=torch.float32, device=context.device)
        patched_context, attention_mask, loc_scale = self._prepare_patched_context(context, context_mask)
        num_context_patches = attention_mask.shape[-1]
        input_embeds = self.input_patch_embedding(patched_context)
        attention_mask = attention_mask.to(self.dtype)
        if self.cfg.use_reg_token:
            reg_input_ids = torch.full((batch_size, 1), self.reg_token_id, dtype=torch.long, device=self.device)
            reg_embeds = self.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, dtype=self.dtype, device=self.device)], dim=-1)
        patched_future, _ = self._prepare_patched_future(loc_scale=loc_scale, num_output_patches=num_output_patches, batch_size=batch_size)
        future_embeds = self.input_patch_embedding(patched_future)
        future_attention_mask = torch.ones(batch_size, num_output_patches, dtype=self.dtype, device=self.device)
        input_embeds = torch.cat([input_embeds, future_embeds], dim=-2)
        attention_mask = torch.cat([attention_mask, future_attention_mask], dim=-1)
        hidden_states = self.encoder(input_embeds, attention_mask=attention_mask)
        expected_seq = num_context_patches + (1 if self.cfg.use_reg_token else 0) + num_output_patches
        assert hidden_states.shape == (batch_size, expected_seq, self.cfg.d_model)
        forecast_embeds = hidden_states[:, -num_output_patches:]
        quantile_preds = self.output_patch_embedding(forecast_embeds)
        B, Nf, QP = quantile_preds.shape
        Q = self.num_quantiles
        P = self.cfg.output_patch_size
        assert QP == Q * P
        quantile_preds_scaled = quantile_preds.view(B, Nf, Q, P).permute(0, 2, 1, 3).reshape(B, Q, Nf * P)
        if bool(constant_fallback_mask.any()):
            fallback_value_for_norm = constant_fallback_value.to(device=quantile_preds_scaled.device, dtype=loc_scale[0].dtype)
            fallback_value_scaled, _ = self.instance_norm(fallback_value_for_norm, loc_scale)
            fallback_value_scaled = fallback_value_scaled.to(quantile_preds_scaled.dtype)
            fallback_preds_scaled = fallback_value_scaled[:, None, :].expand(B, Q, num_output_patches * P)
            quantile_preds_scaled = torch.where(constant_fallback_mask.to(quantile_preds_scaled.device)[:, None, None], fallback_preds_scaled, quantile_preds_scaled)
        H = num_output_patches * P
        quantile_preds_flat = quantile_preds_scaled.reshape(B, Q * H)
        quantile_preds_flat = self.instance_norm.inverse(quantile_preds_flat, loc_scale)
        quantile_preds = quantile_preds_flat.reshape(B, Q, H)
        return quantile_preds
