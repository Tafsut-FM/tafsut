from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TafsutConfig:
    model_type: str = "tafsut"
    architectures: Tuple[str, ...] = ("TafsutModel",)
    context_length: int = 1024
    prediction_length: int = 32
    input_patch_size: int = 32
    output_patch_size: int = 32
    input_patch_stride: int = 32
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)
    use_reg_token: bool = False
    use_arcsinh: bool = True
    time_encoding_scale: Optional[int] = None
    use_constant_series_fallback: bool = True
    constant_atol: float = 1e-6
    constant_rtol: float = 1e-4
    constant_tail_fraction: float = 0.5
    d_model: int = 512
    d_kv: int = 64
    d_ff: int = 1024
    num_layers: int = 6
    num_heads: int = 8
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    feed_forward_proj: str = "relu"
    rope_theta: float = 10000.0
    attn_implementation: str = "sdpa"

