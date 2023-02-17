from abc import abstractmethod
import matplotlib.pyplot as plt
from skimage.transform import resize
from typing import Any
from tqdm.auto import tqdm
import numpy as np
import random 
import logging
import shutil
import os

import rasterio as rio
from rasterio.windows import Window, get_data_window, transform

EMPTY_VALID_DICT = {
    1: 0, 
    0: 0
}


class BaseChete:
    
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
    
    
    @abstractmethod
    def get_window(*args): 
        pass
    
    @abstractmethod
    def get_rasters(*args): 
        pass
    
    
    def get_raster(self, window, no_data_percentage): 
        with rio.open(self.tmp) as src:
            
            transform = src.window_transform(window)

            profile = src.profile
            profile.update({
                'height': self.size,
                'width': self.size,
                'transform': transform})

            new_array = src.read(window=window)
        
        naf = new_array.flatten()
        unique, counts = np.unique(naf, return_counts=True)
        nodata_count = dict(zip(unique, counts)).get(self.nodata, 0)
        
        valid = True if naf.any() and nodata_count/len(naf) <= no_data_percentage else False
        
        return new_array, profile, valid
    
    def save_raster(self, new_array, profile, valid, identifier):
        if valid:  
            self.valid_dict[1]+=1
            try:
                with rio.open(os.path.join(self.output_path, f"croped_{identifier}.tif"), 'w', **profile) as dst:
                        # Read the data from the window and write it to the output raster
                        dst.write(new_array)
            except Exception as e: 
                print("Writing error", e)
        else:
            self.valid_dict[0]+=1
    
    def get_3Ddistribution(self):
        final_counter_array = self.final_counter_array
        bottle_resized = resize(final_counter_array, (300, 300))

        xx, yy =  [], []
        for i, row in enumerate(bottle_resized): 
            xx.append(list(range(len(row))))
            yy.append([i]*len(row))
            
        fig = plt.figure(figsize=(13, 7))
        ax = plt.axes(projection='3d')
        surf = ax.plot_surface(xx, yy, bottle_resized, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Frequency')
        ax.set_title('Surface plot')
        fig.colorbar(surf, shrink=0.5, aspect=5) # add color bar indicating the PDF
        ax.view_init(60, 35)