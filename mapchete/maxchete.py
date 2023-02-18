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

        x_iterations = int(np.floor(abs(self.height-self.size)/stride)) - 1
        y_iterations = int(np.floor(abs(self.width-self.size)/stride)) - 1
        tmp_aux = np.zeros((x_iterations, y_iterations))
        for i in range(x_iterations):
            i_stride = i*stride
            for j in range(y_iterations):
                j_stride = j*stride
                tmp = self.counter_array[i_stride: self.size + i_stride, j_stride: self.size + j_stride]
                tmp_aux[i, j] = tmp.mean()
             
        minimums = np.argwhere(tmp_aux == tmp_aux.min())
        index = int(random.random() * minimums.shape[0])
        min_i, min_j = minimums[index]*stride

        output_array = self.array[min_i: self.size + min_i, min_j: self.size + min_j] 
        no_data_count = output_array[output_array == self.nodata].shape[0]         
        
        # Update density
        if no_data_count/output_length > self.no_data_percentage:
            mask = (output_array == self.nodata)
            self.counter_array[min_i: self.size + min_i, min_j: self.size + min_j][mask] = \
                self.counter_array[min_i: self.size + min_i, min_j: self.size + min_j][mask] + 1
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
                continue
            new_array, profile, valid = self.get_raster(window)
            self.save_raster(new_array, profile, valid, f"{self.identifier}_maxchete_{n_image}")
        
        self.output_mesage()
    