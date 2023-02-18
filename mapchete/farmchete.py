from typing import Union

import rasterio as rio
from rasterio.windows import get_data_window, transform

from .randchete import RandChete
from .seqchete import SeqChete
from .maxchete import MaxChete


class FARMchete():
    
    
    def get(self, crop_type):
        assert crop_type in {"randchete", "seqchete", "maxchete"}
        
        if crop_type == "randchete": 
            return RandChete
        if crop_type == "seqchete":
            return SeqChete
        if crop_type == "maxchete":
            return MaxChete
        
