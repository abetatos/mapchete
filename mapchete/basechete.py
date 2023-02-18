from abc import abstractmethod
from typing import Tuple, Union
import numpy as np
import logging
import shutil
import os

import rasterio as rio
import rasterio.plot as rplt
from rasterio.io import MemoryFile
from rasterio.windows import get_data_window, transform

from skimage.transform import resize
import matplotlib.pyplot as plt


EMPTY_VALID_DICT = {
    1: 0,
    0: 0
}


class BaseChete:

    def __init__(self, filepath: str, log_level: Union[str, int] = logging.DEBUG) -> None:

        self.logger = logging.getLogger("Clip")
        self.logger.setLevel(log_level)

        # Delete dark zones
        self.identifier = os.path.splitext(os.path.basename(filepath))[0]
        with rio.open(filepath) as src:
            profile = src.profile.copy()
            data_window = get_data_window(src.read(masked=True))
            data_transform = transform(data_window, src.transform)
            profile.update(
                transform=data_transform,
                height=data_window.height,
                width=data_window.width
                )

            data = src.read(window=data_window)
            self.array = data[0]

        self.memfile = MemoryFile()
        src = self.memfile.open(**profile)
        src.write(data)

        self.memfile.close()
        self.src = src
        self.width = src.width
        self.height = src.height
        self.nodata = src.nodata

        self.valid_dict = EMPTY_VALID_DICT.copy()

    @abstractmethod
    def get_window(self, window):
        pass

    @abstractmethod
    def _get_rasters(self, n_images: int, size: int, no_data_percentage: float, identifier: str, output_path: str):
        pass

    def get_rasters(self, no_data_percentage: float = 0.2, size: int = 512, n_images: int = 20,
                    output_path: str = "raster_clip", clear_output_path: bool = False):

        self.clear_opath(output_path, clear_output_path)
        self._get_rasters(n_images, size, no_data_percentage, self.identifier, output_path)
        self.output_mesage()

    @staticmethod
    def clear_opath(output_path, clear_output_path):
        if clear_output_path:
            shutil.rmtree(output_path, ignore_errors=True)
        os.makedirs(output_path, exist_ok=True)

    def get_raster_from_window(self, window, size, no_data_percentage):
        transform = self.src.window_transform(window)

        profile = self.src.profile
        profile.update({
            'height': size,
            'width': size,
            'transform': transform})

        new_array = self.src.read(window=window)

        naf = new_array.flatten()
        unique, counts = np.unique(naf, return_counts=True)
        nodata_count = dict(zip(unique, counts)).get(self.nodata, 0)

        valid = True if naf.any() and nodata_count/len(naf) <= no_data_percentage else False

        return new_array, profile, valid

    def save_raster(self, new_array, output_path, profile, valid, identifier):
        if valid:
            self.valid_dict[1] += 1
            try:
                with rio.open(os.path.join(output_path, f"croped_{identifier}.tif"), 'w', **profile) as dst:
                    # Read the data from the window and write it to the output raster
                    dst.write(new_array)
            except Exception as e: 
                self.logger.error("Writing error", e)
        else:
            self.valid_dict[0] += 1

    def check_iteration(self):
        if self.nodata != self.array.max() and self.nodata != self.array.min():
            raise ValueError("Value nodata must be the minimum or maximum value of the input")

        if self.counter_array.min() > 0 and self.distribution_array.max() == 0:
            raise ValueError("Looks like your data is empty")

    def output_mesage(self):
        valid_percentage = round(self.valid_dict[1] / (self.valid_dict[1] + self.valid_dict[0] + 1e-12)*100, 1)
        self.logger.info(f"Process finished with {self.valid_dict[1]} files created"
                         f"which represents {valid_percentage}% of the indicated number.")

    def plot_bands(self):
        rplt.show(self.array, cmap="Greys")

    def plot_hist(self, delete_nodata: bool = True):  
        array = np.where(self.array != self.nodata, self.array, 0) if delete_nodata else self.array
        rplt.show_hist(array)

    def get_3Ddistribution(self, permute: Tuple[int, int] = (1, 0)):
        final_counter_array = np.transpose(self.distribution_array, permute)
        counter_resized = resize(final_counter_array, (300, 300))

        xx, yy = [], []
        for i, row in enumerate(counter_resized): 
            xx.append(list(range(len(row))))
            yy.append([i]*len(row))

        fig = plt.figure(figsize=(13, 7))
        ax = plt.axes(projection='3d')
        surf = ax.plot_surface(xx, yy, counter_resized, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Frequency')
        ax.set_title('Surface plot')
        fig.colorbar(surf, shrink=0.5, aspect=5)  # add color bar indicating the PDF
        ax.view_init(60, 35)

        return fig, ax
