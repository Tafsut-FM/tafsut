from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


def _as_numpy(values: torch.Tensor | np.ndarray | Sequence[float]) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _select_series(values: np.ndarray, series_index: int, expected_ndim: tuple[int, ...]) -> np.ndarray:
    if values.ndim not in expected_ndim:
        raise ValueError(f"unexpected array shape {values.shape}")
    if values.ndim == min(expected_ndim):
        return values
    if not 0 <= series_index < values.shape[0]:
        raise IndexError(f"series_index={series_index} is outside batch size {values.shape[0]}")
    return values[series_index]


def _nearest_quantile_index(quantiles: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(quantiles - value)))


def plot_forecast(
    context: torch.Tensor | np.ndarray | Sequence[float],
    prediction: torch.Tensor | np.ndarray,
    quantiles: Sequence[float],
    *,
    target: torch.Tensor | np.ndarray | Sequence[float] | None = None,
    series_index: int = 0,
    history_length: int | None = None,
    interval_levels: Iterable[float] = (0.5, 0.8, 0.9),
    title: str = "Tafsut forecast",
    ax=None,
):
    """Plot history, median forecast, central quantile intervals, and optional target.

    Parameters
    ----------
    context:
        Historical values with shape ``(T,)`` or ``(B, T)``.
    prediction:
        Forecast tensor with shape ``(H, Q)`` or ``(B, H, Q)``.
    quantiles:
        Quantile levels corresponding to the last prediction dimension.
    target:
        Optional future observations with shape ``(H,)`` or ``(B, H)``.
    series_index:
        Batch item to visualize when batched inputs are supplied.
    history_length:
        Number of historical points to display. ``None`` shows all context.
    interval_levels:
        Central probability masses to shade, e.g. ``0.8`` requests the
        interval between the 0.1 and 0.9 quantiles (nearest available levels).
    title:
        Plot title.
    ax:
        Existing matplotlib axes. A new figure is created when omitted.

    Returns
    -------
    (figure, axes)
        Matplotlib figure and axes objects.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Visualization requires matplotlib. Install it with `pip install matplotlib`."
        ) from exc

    history = _select_series(_as_numpy(context), series_index, (1, 2)).astype(float, copy=False)
    pred = _select_series(_as_numpy(prediction), series_index, (2, 3)).astype(float, copy=False)

    if pred.ndim != 2:
        raise ValueError(f"prediction must resolve to shape (H, Q), got {pred.shape}")

    q = np.asarray(quantiles, dtype=float)
    if q.ndim != 1 or q.size != pred.shape[-1]:
        raise ValueError(
            f"quantiles must have length {pred.shape[-1]}, got shape {q.shape}"
        )
    if np.any(~np.isfinite(q)) or np.any((q < 0.0) | (q > 1.0)):
        raise ValueError("quantiles must be finite values in [0, 1]")

    if history_length is not None:
        history_length = int(history_length)
        if history_length <= 0:
            raise ValueError("history_length must be positive")
        history = history[-history_length:]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    history_x = np.arange(-history.size, 0)
    future_x = np.arange(pred.shape[0])

    ax.plot(history_x, history, label="History")

    median_idx = _nearest_quantile_index(q, 0.5)
    ax.plot(future_x, pred[:, median_idx], label=f"Forecast q={q[median_idx]:g}")

    # Draw wider intervals first so narrower bands remain visible.
    plotted_pairs: set[tuple[int, int]] = set()
    levels = sorted({float(level) for level in interval_levels}, reverse=True)
    for level in levels:
        if not 0.0 < level < 1.0:
            raise ValueError("interval_levels values must be strictly between 0 and 1")
        tail = (1.0 - level) / 2.0
        low_idx = _nearest_quantile_index(q, tail)
        high_idx = _nearest_quantile_index(q, 1.0 - tail)
        if low_idx == high_idx or q[low_idx] >= q[high_idx]:
            continue
        pair = (low_idx, high_idx)
        if pair in plotted_pairs:
            continue
        plotted_pairs.add(pair)
        ax.fill_between(
            future_x,
            pred[:, low_idx],
            pred[:, high_idx],
            alpha=0.18,
            label=f"q={q[low_idx]:g}–{q[high_idx]:g}",
        )

    if target is not None:
        future = _select_series(_as_numpy(target), series_index, (1, 2)).astype(float, copy=False)
        future = future[: pred.shape[0]]
        ax.plot(np.arange(future.size), future, label="Observed future")

    ax.axvline(-0.5, linewidth=1, linestyle="--", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Time relative to forecast origin")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def save_forecast_plot(
    path: str | Path,
    context: torch.Tensor | np.ndarray | Sequence[float],
    prediction: torch.Tensor | np.ndarray,
    quantiles: Sequence[float],
    **plot_kwargs,
) -> Path:
    """Create a forecast plot and save it to PNG, PDF, SVG, or another matplotlib format."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Visualization requires matplotlib. Install it with `pip install matplotlib`."
        ) from exc

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, _ = plot_forecast(context, prediction, quantiles, **plot_kwargs)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output
