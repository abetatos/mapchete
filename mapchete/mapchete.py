from mapchete import RandChete, SeqChete, MaxChete

from typing import Any
import logging
import shutil
import os

import rasterio as rio
from rasterio.windows import get_data_window, transform

EMPTY_VALID_DICT = {
    1: 0, 
    0: 0
}

class MAPchete():
    
    def __init__(self, filepath: str, size: int=512, output_path: str="raster_clip", clear_output_path: bool=True, log_level: Any[str, int]=logging.DEBUG) -> None:        
        """_summary_

        Args:
            filepath (_type_): _description_
            size (int, optional): _description_. Defaults to 512.
            output_path (str, optional): _description_. Defaults to "raster_clip".
            clear_output_path (bool, optional): _description_. Defaults to True.
            log_level (_type_, optional): _description_. Defaults to logging.DEBUG.
        """
        self.logger = logging.getLogger("Clip")
        self.logger.setLevel(log_level)
        
        # Clean 
        with rio.open(filepath) as src:
            profile = src.profile.copy()
            data_window = get_data_window(src.read(masked=True))
            data_transform = transform(data_window, src.transform)
            profile.update(
                transform=data_transform,
                height=data_window.height,
                width=data_window.width)

            data = src.read(window=data_window)
            self.array = data[0]
            
        self.tmp = os.path.join(output_path, "tmp.tiff")
        with rio.open(self.tmp, 'w', **profile) as dst:
            dst.write(data)
            
        with rio.open(self.tmp) as src:
             
            self.rasterfile = src
        
            self.width = self.rasterfile.width
            self.height = self.rasterfile.height
            self.nodata = self.rasterfile.nodata
        
        self.size = size
        self.output_path = output_path
        if clear_output_path: 
            shutil.rmtree(output_path, ignore_errors=True)
        
        self.crop_function = None
        self.valid_dict = EMPTY_VALID_DICT.copy()

    def get(crop_type):
        assert crop_type in {"random", "sequential", "maxchete"}
        
        if crop_type == "random": 
            return RandChete
        if crop_type == "sequential":
            return SeqChete
        if crop_type == "maxchete":
            return MaxChete
        
