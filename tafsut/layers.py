from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ACT2FN = {'relu': F.relu, 'gelu': F.gelu, 'silu': F.silu, 'swish': F.silu, 'tanh': torch.tanh}

class InstanceNorm(nn.Module):

    def __init__(self, eps: float=1e-06, use_arcsinh: bool=False) -> None:
        super().__init__()
        self.eps = eps
        self.use_arcsinh = use_arcsinh

    def forward(self, x: torch.Tensor, loc_scale: tuple[torch.Tensor, torch.Tensor] | None=None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if loc_scale is None:
            loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num((x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0)
        else:
            loc, scale = loc_scale
            loc = loc.to(torch.float32)
            scale = scale.to(torch.float32)
        scale = torch.where(scale == 0, self.eps, scale)
        scaled_x = (x - loc) / scale
        if self.use_arcsinh:
            scaled_x = torch.arcsinh(scaled_x)
        return (scaled_x.to(orig_dtype), (loc, scale))

    def inverse(self, x: torch.Tensor, loc_scale: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        loc, scale = loc_scale
        if self.use_arcsinh:
            x = torch.sinh(x)
        x = x * scale + loc
        return x.to(orig_dtype)

class Patch(nn.Module):

    def __init__(self, patch_size: int, patch_stride: int):
        super().__init__()
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)

    @staticmethod
    def _pad_len(T: int, P: int, S: int) -> int:
        if T < P:
            return P - T
        rem = (T - P) % S
        return 0 if rem == 0 else S - rem

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f'Patch expects (B, T), got {tuple(x.shape)}')
        B, T = x.shape
        P, S = (self.patch_size, self.patch_stride)
        if T <= 0:
            raise ValueError('Sequence length must be > 0')
        pad = self._pad_len(T, P, S)
        if pad > 0:
            padding = torch.full((B, pad), float('nan'), dtype=x.dtype, device=x.device)
            x = torch.cat([padding, x], dim=-1)
        return x.unfold(dimension=-1, size=P, step=S)

class RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states

class RoPE(nn.Module):

    def __init__(self, dim: int, base: float=10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f'RoPE head dim must be even, got {dim}')
        self.dim = int(dim)
        self.base = float(base)
        inv_freq = 1.0 / self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if x.device.type != 'mps' else 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return (cos.to(dtype=x.dtype), sin.to(dtype=x.dtype))

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int=1) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = q * cos + RoPE.rotate_half(q) * sin
        k_embed = k * cos + RoPE.rotate_half(k) * sin
        return (q_embed, k_embed)

class MLP(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout_rate: float, dense_act_fn: str):
        super().__init__()
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = ACT2FN[dense_act_fn]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states

class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout_rate: float, dense_act_fn: str, layer_norm_epsilon: float):
        super().__init__()
        self.mlp = MLP(d_model, d_ff, dropout_rate, dense_act_fn)
        self.layer_norm = RMSNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.mlp(forwarded_states)
        return hidden_states + self.dropout(forwarded_states)

class MHA(nn.Module):

    def __init__(self, d_model: int, d_kv: int, num_heads: int, dropout_rate: float, rope_theta: float, use_rope: bool=True, attn_implementation: str='sdpa'):
        super().__init__()
        self.d_model = int(d_model)
        self.kv_proj_dim = int(d_kv)
        self.n_heads = int(num_heads)
        self.dropout = float(dropout_rate)
        self.inner_dim = self.n_heads * self.kv_proj_dim
        self.attn_implementation = attn_implementation
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)
        self.use_rope = use_rope
        if use_rope:
            self.rope_embed = RoPE(dim=self.kv_proj_dim, base=rope_theta)

    def _eager_attention(self, query_states: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(query_states, key_states.transpose(3, 2))
        scores = scores + mask
        weights = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        return torch.matmul(weights, value_states)

    def _sdpa_attention(self, query_states: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0, scale=1.0)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor, position_ids: Optional[torch.Tensor]=None) -> torch.Tensor:
        if self.use_rope:
            assert position_ids is not None, 'position_ids must be provided when use_rope=True'
        batch_size, seq_length, _ = hidden_states.shape

        def shape(states: torch.Tensor) -> torch.Tensor:
            return states.view(batch_size, seq_length, self.n_heads, self.kv_proj_dim).transpose(1, 2)

        def unshape(states: torch.Tensor) -> torch.Tensor:
            return states.transpose(1, 2).contiguous().view(batch_size, seq_length, self.inner_dim)
        query_states = shape(self.q(hidden_states))
        key_states = shape(self.k(hidden_states))
        value_states = shape(self.v(hidden_states))
        if self.use_rope:
            cos, sin = self.rope_embed(value_states, position_ids)
            query_states, key_states = RoPE.apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if self.attn_implementation == 'sdpa' and hasattr(F, 'scaled_dot_product_attention'):
            attn_output = self._sdpa_attention(query_states, key_states, value_states, mask)
        else:
            attn_output = self._eager_attention(query_states, key_states, value_states, mask)
        attn_output = unshape(attn_output)
        attn_output = self.o(attn_output)
        return attn_output

class TimeSelfAttention(nn.Module):

    def __init__(self, d_model: int, d_kv: int, num_heads: int, dropout_rate: float, rope_theta: float, layer_norm_epsilon: float, attn_implementation: str):
        super().__init__()
        self.self_attention = MHA(d_model=d_model, d_kv=d_kv, num_heads=num_heads, dropout_rate=dropout_rate, rope_theta=rope_theta, use_rope=True, attn_implementation=attn_implementation)
        self.layer_norm = RMSNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        normed = self.layer_norm(hidden_states)
        attn_out = self.self_attention(normed, mask=attention_mask, position_ids=position_ids)
        return hidden_states + self.dropout(attn_out)

class ResidualBlock(nn.Module):

    def __init__(self, in_dim: int, h_dim: int, out_dim: int, act_fn_name: str, dropout_p: float=0.0, use_layer_norm: bool=False) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.act = ACT2FN[act_fn_name]
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = RMSNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x)
        out = out + res
        if self.use_layer_norm:
            return self.layer_norm(out)
        return out

class EncoderBlock(nn.Module):

    def __init__(self, d_model: int, d_kv: int, num_heads: int, d_ff: int, dropout_rate: float, dense_act_fn: str, rope_theta: float, layer_norm_epsilon: float, attn_implementation: str):
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(TimeSelfAttention(d_model=d_model, d_kv=d_kv, num_heads=num_heads, dropout_rate=dropout_rate, rope_theta=rope_theta, layer_norm_epsilon=layer_norm_epsilon, attn_implementation=attn_implementation))
        self.layer.append(FeedForward(d_model=d_model, d_ff=d_ff, dropout_rate=dropout_rate, dense_act_fn=dense_act_fn, layer_norm_epsilon=layer_norm_epsilon))

    def forward(self, hidden_states: torch.Tensor, *, position_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.layer[0](hidden_states, attention_mask=attention_mask, position_ids=position_ids)
        hidden_states = self.layer[1](hidden_states)
        return hidden_states

class Encoder(nn.Module):

    def __init__(self, num_layers: int, d_model: int, d_kv: int, num_heads: int, d_ff: int, dropout_rate: float, dense_act_fn: str, rope_theta: float, layer_norm_epsilon: float, attn_implementation: str):
        super().__init__()
        self.block = nn.ModuleList([EncoderBlock(d_model=d_model, d_kv=d_kv, num_heads=num_heads, d_ff=d_ff, dropout_rate=dropout_rate, dense_act_fn=dense_act_fn, rope_theta=rope_theta, layer_norm_epsilon=layer_norm_epsilon, attn_implementation=attn_implementation) for _ in range(num_layers)])
        self.final_layer_norm = RMSNorm(d_model, eps=layer_norm_epsilon)
        self.dropout = nn.Dropout(dropout_rate)

    @staticmethod
    def _expand_and_invert_time_attention_mask(attention_mask: torch.Tensor, floating_type: torch.dtype) -> torch.Tensor:
        assert attention_mask.ndim == 2, 'attention_mask must have shape (batch, seq_len)'
        attention_mask = attention_mask[:, None, None, :].to(dtype=floating_type)
        attention_mask = (1.0 - attention_mask) * torch.finfo(floating_type).min
        return attention_mask

    def forward(self, inputs_embeds: torch.Tensor, *, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None) -> torch.Tensor:
        batch_size, seq_length = inputs_embeds.size()[:-1]
        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0)
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        extended_attention_mask = self._expand_and_invert_time_attention_mask(attention_mask, inputs_embeds.dtype)
        hidden_states = self.dropout(inputs_embeds)
        for layer_module in self.block:
            hidden_states = layer_module(hidden_states, position_ids=position_ids, attention_mask=extended_attention_mask)
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
