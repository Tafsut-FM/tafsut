from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import torch

from .config import TafsutConfig
from .model import TafsutModel


def load_config(path: str | Path) -> TafsutConfig:
    path = Path(path)
    data = json.loads(path.read_text())
    allowed = {field.name for field in fields(TafsutConfig)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown config fields: {unknown}")
    if "quantiles" in data:
        data["quantiles"] = tuple(float(q) for q in data["quantiles"])
    if "architectures" in data:
        data["architectures"] = tuple(str(name) for name in data["architectures"])
    return TafsutConfig(**data)


def save_config(config: TafsutConfig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(config)
    data["quantiles"] = list(config.quantiles)
    data["architectures"] = list(config.architectures)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return path


def _load_torch_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        obj: Any = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or not obj or not all(
        isinstance(k, str) and torch.is_tensor(v) for k, v in obj.items()
    ):
        raise ValueError(f"Expected a raw tensor state_dict in {path}")
    return obj


def _resolve_source(
    source: str | Path,
    *,
    revision: str | None = None,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
    path = Path(source).expanduser()
    if path.exists():
        if not path.is_dir():
            raise ValueError("Model source must be a directory containing config.json and weights")
        return path

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("Install huggingface_hub to load a remote model repository") from exc

    return Path(
        snapshot_download(
            repo_id=str(source),
            revision=revision,
            token=token,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=local_files_only,
            allow_patterns=["config.json", "model.safetensors", "pytorch_model.bin"],
        )
    )


def load_tafsut(
    source: str | Path,
    device: str | torch.device | None = None,
    *,
    revision: str | None = None,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> TafsutModel:
    root = _resolve_source(
        source,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing {config_path}")

    cfg = load_config(config_path)
    model = TafsutModel(cfg)

    safe_path = root / "model.safetensors"
    bin_path = root / "pytorch_model.bin"
    if safe_path.is_file():
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("Install safetensors to load model.safetensors") from exc
        state_dict = load_file(str(safe_path), device="cpu")
    elif bin_path.is_file():
        state_dict = _load_torch_state(bin_path)
    else:
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {root}")

    model.load_state_dict(state_dict, strict=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    model.to(device)
    model.eval()
    return model


def save_tafsut(
    model: TafsutModel,
    directory: str | Path,
    *,
    safe_serialization: bool = True,
) -> Path:
    root = Path(directory).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    save_config(model.cfg, root / "config.json")
    state_dict = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
    }

    if safe_serialization:
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise ImportError("Install safetensors to save model.safetensors") from exc
        weights_path = root / "model.safetensors"
        save_file(state_dict, str(weights_path), metadata={"format": "pt"})
    else:
        weights_path = root / "pytorch_model.bin"
        torch.save(state_dict, weights_path)

    return weights_path
