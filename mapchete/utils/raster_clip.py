

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
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


class MAPchete: 
    
    def __init__(self, filepath, size=512, output_path = "raster_clip", clear_output_path=True) -> None:        
        
        self.logger = logging.getLogger("Clip")
        self.logger.setLevel(logging.DEBUG)
        
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
            
        
        with rio.open(os.path.join(output_path, "tmp.tiff"), 'w', **profile) as dst:
            dst.write(data)
            
        with rio.open("tmp.tiff") as src:
             
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
        
        # fig, axes = plt.subplots(10, 10, figsize=(10,10))
        
        # self.axes = (ax for ax in axes.flat)
    
    def get_random_window(self): 
        xmin, xmax = 0, self.height - self.size
        ymin, ymax = 0, self.width - self.size
        i, j = random.randint(xmin, xmax), random.randint(ymin, ymax)
        return i, j, Window(row_off=i, col_off=j, width=self.size, height=self.size)

    def get_sequential_window(self, i, j): 
        return Window(row_off=i, col_off=j, width=self.size, height=self.size)
    
    
    def get_maximal_window(self, no_data_percentage): 
        
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
                # above
            # else: 
            #     for j in range(y_iterations):
            #         j_stride = min(j*stride, self.counter_array.shape[1]-1)
            #         try: 
            #             tmp_aux[i, j] += self.counter_array[i_stride: self.size + i_stride, j_stride + self.size: j_stride + self.size+1].sum()  -\
            #                              self.counter_array[i_stride: self.size + i_stride, j_stride: j_stride+1].sum()
                                        
            #         except Exception as e: 
            #             print(j_stride, self.counter_array.shape[1])
            #             raise e
        
        minimums = np.argwhere(tmp_aux == tmp_aux.min())
        index = int(random.random() * minimums.shape[0])
        min_i, min_j = minimums[index]*stride

        output_array = self.array[min_i: self.size + min_i, min_j: self.size + min_j] 
        no_data_count = output_array[output_array == self.nodata].shape[0]         
        # Update density

        # try: 
        #     next(self.axes).imshow(self.counter_array)
        # except StopIteration: 
        #     pass

        if no_data_count/output_length > no_data_percentage:
            
            mask = (output_array == self.nodata)
            self.counter_array[min_i: self.size + min_i, min_j: self.size + min_j][mask] = \
                self.counter_array[min_i: self.size + min_i, min_j: self.size + min_j][mask] + 1
            
            
            # counter_array_aux = self.counter_array[min_i: self.size + min_i, min_j: self.size + min_j]
            # nodata_value = counter_array_aux.mean() + 1
            # self.counter_array[min_i: self.size + min_i, min_j: self.size + min_j]  = np.where(~(output_array==self.nodata),
            #                                                                                    counter_array_aux,
            #                                                                                    nodata_value)
            
            
            # for i, row in enumerate(output_array):
            #     for j, item in enumerate(row):
            #         if item == self.nodata: 
            #             self.counter_array[i + min_i, j + min_j] += 1

            return None, None
        
        else: 
            self.counter_array[min_i: self.size + min_i, min_j: self.size + min_j] += 1
            self.final_counter_array[min_i: self.size + min_i, min_j: self.size + min_j] += 1
            
            return min_i, min_j
            
                
    def get_raster(self, window, no_data_percentage): 
        with rio.open("tmp.tiff") as src:
            
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
    
       
    def get_rasters(self, crop_type, no_data_percentage=0.2, n_images=20, identifier = ""): 
        assert crop_type in {"random", "sequential", "maxchete"}
        
        self.valid_dict = EMPTY_VALID_DICT.copy()
        
        if crop_type == "random": 
            self.final_counter_array = np.zeros_like(self.array)
            for i in tqdm(range(1, n_images+1)): 
                i, j, window = self.get_random_window()
            
                new_array, profile, valid = self.get_raster(window, no_data_percentage)
                if valid: 
                    self.final_counter_array[i: self.size + i, j: self.size + j] += 1
                self.save_raster(new_array, profile, valid, f"{identifier}_random_{i}")
        elif crop_type == "sequential":             
            for i in range(int(np.ceil(self.height/self.size))): 
                for j in range(int(np.ceil(self.width/self.size))):
                    
                    window = self.get_sequential_window(i*self.size, j*self.size)
                    new_array, profile, valid = self.get_raster(window, no_data_percentage)
                    self.save_raster(new_array, profile, valid, f"{identifier}_sequential_{i}_{j}")
        elif crop_type == "maxchete":                         
            self.counter_array = np.zeros_like(self.array)
            self.final_counter_array = np.zeros_like(self.array)
            
            for n_image in tqdm(range(n_images)):
               
                i, j = self.get_maximal_window(no_data_percentage)
                
                if i is None and j is None:
                    continue
                
                window = self.get_sequential_window(i, j)
                new_array, profile, valid = self.get_raster(window, no_data_percentage)
                self.save_raster(new_array, profile, valid, f"{identifier}_maximize_area_{n_image}")

        self.logger.warning(f"Valid results {self.valid_dict}")
        # print(f"Valid results {self.valid_dict}")