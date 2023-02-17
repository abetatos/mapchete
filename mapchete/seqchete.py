import random

import numpy as np
from tqdm.auto import tqdm
from rasterio.windows import Window

from mapchete import BaseChete


def SeqChete(BaseChete): 
    
    def get_window(self, i, j):
        return Window(row_off=i, col_off=j, width=self.size, height=self.size)
    
   
    def get_rasters(self, no_data_percentage=0.2, n_images=20, identifier = ""):
        
        images_done = 0
        i_steps = int(np.ceil(self.height/self.size))
        j_steps = int(np.ceil(self.width/self.size))
        while True:
            for i in range(i_steps): 
                for j in range(j_steps):
                    window = self.get_sequential_window(i*self.size, j*self.size)
                    new_array, profile, valid = self.get_raster(window, no_data_percentage)
                    self.save_raster(new_array, profile, valid, f"{identifier}_sequential_{i}_{j}")
                    if images_done + 1 == n_images: 
                        break
                    else: 
                        images_done += 1
                else: 
                    continue
                break
            else: 
                continue
            break
                
            