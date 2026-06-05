__author__ = "Alejandro Betato Sancerni"
__author_email__ = "abetatos@gmail.com"
__version__ = "0.1.0"

from .core import Tiler
from .farmchete import FARMchete
from .strategies import (
    SamplingStrategy,
    RandomStrategy,
    MaxCoverage,
    InfoCoverage,
    PoissonDisk,
    SlidingWindow,
    get_strategy,
)
from .utils.merge_tiffs import merge_tiffs

__all__ = [
    "Tiler",
    "FARMchete",
    "SamplingStrategy",
    "RandomStrategy",
    "MaxCoverage",
    "InfoCoverage",
    "PoissonDisk",
    "SlidingWindow",
    "get_strategy",
    "merge_tiffs",
]
