"""Backwards-compatible factory.

``FARMchete`` predates the :class:`~mapchete.core.Tiler` API and is kept so older
code (and the example notebook) keeps working. New code should prefer
``Tiler.from_name(path, "maxchete")`` or ``Tiler(path, MaxCoverage())``.
"""
from .core import Tiler


class FARMchete:
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath

    def get(self, crop_type: str) -> Tiler:
        return Tiler.from_name(self.filepath, crop_type)
