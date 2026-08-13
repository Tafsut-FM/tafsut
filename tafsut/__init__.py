from importlib.metadata import PackageNotFoundError, version

from .checkpoint import load_config, load_tafsut, save_config, save_tafsut
from .config import TafsutConfig
from .model import TafsutModel
from .inference import forecast
from .visualization import plot_forecast, save_forecast_plot

try:
    __version__ = version("tafsut")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "TafsutConfig",
    "TafsutModel",
    "forecast",
    "load_config",
    "load_tafsut",
    "save_config",
    "save_tafsut",
    "plot_forecast",
    "save_forecast_plot",
    "__version__",
]
