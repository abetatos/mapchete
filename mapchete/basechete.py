import os
from abc import abstractmethod

import rasterio as rio
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np


class BaseChete:
    
    
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
            
    def output_mesage(self): 
        valid_percentage = round(self.valid_dict[1] / (self.valid_dict[1] + self.valid_dict[0])*100, 1)
        self.logger.info(f"Process finished with {self.valid_dict[1]} files created"
                         f"which represents {valid_percentage}% of the indicated number.")
    
    def check_iteration(self): 
        
        if self.nodata != self.array.max() and self.nodata != self.array.min():
            raise ValueError("Value nodata must be the minimum or maximum value of the input")
        
        if self.counter_array.min() > 0  and self.final_counter_array.max() == 0: 
            raise ValueError("Looks like your data is empty")

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