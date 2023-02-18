import random

import numpy as np
from tqdm import tqdm
from rasterio.windows import Window

from .basechete import BaseChete


class MaxChete(BaseChete): 
    
    # def __init__(self, *args, **kwargs): 
    #     super().__init__(*args, **kwargs)
        
    def get_window(self):        
        output_length = self.size**2
        
        stride = int(max(self.size/10, random.random()*(self.size-1)))

        x_iterations = int(np.floor(abs(self.height-self.size)/stride))
        y_iterations = int(np.floor(abs(self.width-self.size)/stride))
        
        tmp_aux = np.zeros((x_iterations, y_iterations))
        for i in range(x_iterations):
            i_init = i*stride
            i_end = i*stride + self.size
            for j in range(y_iterations):
                j_init = j*stride
                j_end = j*stride + self.size
                if j == 0:
                    tmp = self.counter_array[i_init: i_end, j_init: j_end]
                    tmp_aux[i, j] = tmp.sum()
                else: # Just for optimization purposes. Reduction of ~20% in execution time
                    old_value = tmp_aux[i, j-1]
                    delta = (
                        - self.counter_array[i_init: i_end, (j-1)*stride : j_init].sum() \
                        + self.counter_array[i_init: i_end, (j-1)*stride + self.size: j_end].sum()
                    )
                    tmp_aux[i, j] = old_value + delta
                        
        minimums = np.argwhere(tmp_aux == tmp_aux.min())
        index = int(random.random() * minimums.shape[0])
        min_i, min_j = minimums[index]*stride

        output_array = self.array[min_i: self.size + min_i, min_j: self.size + min_j] 
        no_data_count = output_array[output_array == self.nodata].shape[0]         
        
        # Update density
        if no_data_count/output_length > self.no_data_percentage:
            mask = (output_array == self.nodata)
            self.counter_array[min_i: self.size + min_i, min_j: self.size + min_j][mask] +=1 
            return None
        else: 
            self.counter_array[min_i: self.size + min_i, min_j: self.size + min_j] += 1
            self.final_counter_array[min_i: self.size + min_i, min_j: self.size + min_j] += 1
            return Window(min_j, min_i, self.size, self.size)
        
        
    def get_rasters(self):
        self.counter_array = np.zeros_like(self.array)
        self.final_counter_array = np.zeros_like(self.array)

        for n_image in tqdm(range(self.n_images)):
            
            window = self.get_window()
            if window is None:
                self.valid_dict[0] +=1
                continue
            new_array, profile, valid = self.get_raster(window)
            self.save_raster(new_array, profile, valid, f"{self.identifier}_maxchete_{n_image}")
        
        self.output_mesage()
    