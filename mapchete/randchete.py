import random

import numpy as np
from tqdm.auto import tqdm
from rasterio.windows import Window

from .basechete import BaseChete


class RandChete(BaseChete): 
    
    def get_window(self):
        xmin, xmax = 0, self.height - self.size
        ymin, ymax = 0, self.width - self.size
        i, j = random.randint(xmin, xmax), random.randint(ymin, ymax)
        return i, j, Window(row_off=i, col_off=j, width=self.size, height=self.size)
    
    def get_rasters(self, no_data_percentage=0.2, n_images=20, identifier = ""):
        
        self.final_counter_array = np.zeros_like(self.array)
        
        for i in tqdm(range(1, n_images+1)): 
            i, j, window = self.get_window()
            new_array, profile, valid = self.get_raster(window)
            if valid: 
                self.final_counter_array[i: self.size + i, j: self.size + j] += 1
            self.save_raster(new_array, profile, valid, f"{identifier}_randchete_{i}")
        
        self.output_mesage()
