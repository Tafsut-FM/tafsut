from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_tafsut
from .model import TafsutModel


def forecast(
    model: TafsutModel,
    context: torch.Tensor | np.ndarray,
    horizon: int | None = None,
    context_mask: torch.Tensor | np.ndarray | None = None,
) -> torch.Tensor:
    """Generate probabilistic forecasts.

    Parameters
    ----------
    model:
        Loaded Tafsut model.
    context:
        Historical values with shape ``(T,)`` or ``(B, T)``.
    horizon:
        Number of future time steps to forecast. Defaults to the model's
        configured prediction length.
    context_mask:
        Optional boolean mask matching ``context``. Missing values may also be
        represented directly by NaN values in ``context``.

    Returns
    -------
    torch.Tensor
        Forecasts on CPU with shape ``(B, horizon, Q)``, where ``Q`` is the
        number of configured quantiles.
    """
    device = next(model.parameters()).device
    x = torch.as_tensor(context, dtype=torch.float32, device=device)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2:
        raise ValueError(f"context must have shape (T,) or (B, T), got {tuple(x.shape)}")

    if context_mask is None:
        mask = torch.isfinite(x)
    else:
        mask = torch.as_tensor(context_mask, dtype=torch.bool, device=device)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape != x.shape:
            raise ValueError(
                f"context_mask shape {tuple(mask.shape)} does not match context {tuple(x.shape)}"
            )
        mask = mask & torch.isfinite(x)

    if horizon is None:
        horizon = int(model.cfg.prediction_length)
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    quantiles = torch.tensor(model.cfg.quantiles, dtype=torch.float32)
    median_index = int(torch.argmin(torch.abs(quantiles - 0.5)).item())
    chunks: list[torch.Tensor] = []
    remaining = horizon

    model.eval()
    with torch.inference_mode():
        while remaining > 0:
            block = model(x, context_mask=mask).transpose(1, 2)
            take = min(remaining, block.shape[1])
            current = block[:, :take, :]
            chunks.append(current)

            if take < remaining:
                feedback = current[:, :, median_index]
                x = torch.cat([x, feedback], dim=-1)
                mask = torch.cat(
                    [mask, torch.ones_like(feedback, dtype=torch.bool)], dim=-1
                )
                max_context = int(model.cfg.context_length)
                if x.shape[-1] > max_context:
                    x = x[..., -max_context:]
                    mask = mask[..., -max_context:]

            remaining -= take

    return torch.cat(chunks, dim=1).cpu()


def _parse_context(text: str) -> np.ndarray:
    values = []
    for token in text.split(","):
        token = token.strip()
        values.append(
            float("nan") if token.lower() in {"nan", "na", "null"} else float(token)
        )
    return np.asarray(values, dtype=np.float32)


def _load_array_file(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".json":
        return np.asarray(json.loads(path.read_text()), dtype=np.float32)
    raise ValueError("array file must be .npy or .json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Tafsut forecasting from a local folder or Hugging Face repository"
    )
    parser.add_argument(
        "--model",
        default="Tafsut-FM/tafsut-univariate-base",
        help="Local model directory or Hugging Face repository ID",
    )
    parser.add_argument(
        "--context", help="Comma-separated values; use nan for missing observations"
    )
    parser.add_argument("--context-file", type=Path, help=".npy or .json context array")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--plot-output", type=Path, help="Save forecast visualization (.png, .pdf, .svg, ...)"
    )
    parser.add_argument(
        "--show-plot", action="store_true", help="Display the forecast visualization interactively"
    )
    parser.add_argument(
        "--target-file", type=Path, help="Optional .npy or .json future observations to overlay"
    )
    parser.add_argument("--series-index", type=int, default=0, help="Batch series to visualize")
    parser.add_argument(
        "--history-length", type=int, default=None, help="Historical points shown in the plot"
    )
    parser.add_argument("--plot-title", default="Tafsut forecast")
    args = parser.parse_args()

    if (args.context is None) == (args.context_file is None):
        parser.error("Provide exactly one of --context or --context-file")

    context = (
        _parse_context(args.context)
        if args.context is not None
        else _load_array_file(args.context_file)
    )
    model = load_tafsut(args.model, device=args.device)
    pred = forecast(model, context, horizon=args.horizon)

    print(json.dumps({"quantiles": list(model.cfg.quantiles), "forecast": pred.tolist()}))

    if args.plot_output is not None or args.show_plot:
        from .visualization import plot_forecast

        target = _load_array_file(args.target_file) if args.target_file is not None else None
        fig, _ = plot_forecast(
            context,
            pred,
            model.cfg.quantiles,
            target=target,
            series_index=args.series_index,
            history_length=args.history_length,
            title=args.plot_title,
        )
        if args.plot_output is not None:
            args.plot_output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.plot_output, dpi=160, bbox_inches="tight")
        if args.show_plot:
            import matplotlib.pyplot as plt

            plt.show()
        else:
            import matplotlib.pyplot as plt

            plt.close(fig)


if __name__ == "__main__":
    main()
