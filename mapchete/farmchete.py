from typing import Union

import rasterio as rio
from rasterio.windows import get_data_window, transform

from .randchete import RandChete
from .seqchete import SeqChete
from .maxchete import MaxChete


class FARMchete():

    def __init__(self, filepath: str) -> None:
        """Farm method to return the desired class, pases the filename to the functions to instantiate them

        Args:
            filepath (str): Input filepath
        """
        self.filepath = filepath

    def get(self, crop_type: str):
        assert crop_type in {"randchete", "seqchete", "maxchete"}

        if crop_type == "randchete":
            return RandChete(self.filepath)
        if crop_type == "seqchete":
            return SeqChete(self.filepath)
        if crop_type == "maxchete":
            return MaxChete(self.filepath)
