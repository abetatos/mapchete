import numpy as np
from rasterio.windows import Window

from .basechete import BaseChete


class SeqChete(BaseChete): 
    
    def get_window(self, i, j):
        return Window(row_off=i, col_off=j, width=self.size, height=self.size)
    
    def get_rasters(self, n_images=20, identifier = ""):
        
        self.final_counter_array = np.zeros_like(self.array)
        
        images_done = 0
        i_steps = int(np.ceil(self.height/self.size))
        j_steps = int(np.ceil(self.width/self.size))
        while True:
            for i in range(i_steps): 
                for j in range(j_steps):
                    self.final_counter_array[i*self.size: (i+1)*self.size, j*self.size: (j+1)*self.size] += 1
                    window = self.get_window(i*self.size, j*self.size)
                    new_array, profile, valid = self.get_raster(window)
                    self.save_raster(new_array, profile, valid, f"{identifier}_seqchete_{images_done}")
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
        
        self.output_mesage()
        
                
            