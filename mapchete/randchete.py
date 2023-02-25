import random
from typing import Tuple

import numpy as np
from tqdm.auto import tqdm
from rasterio.windows import Window

from .basechete import BaseChete


class RandChete(BaseChete):

    def get_window(self, size: int) -> Tuple[int, int, Window]:
        """Generates a randow window between the image limits

        Args:
            size (int): Output file size

        Returns:
            Tuple[int, int, Window]: Returns indices and window
        """
        xmin, xmax = 0, self.height - size
        ymin, ymax = 0, self.width - size
        i, j = random.randint(xmin, xmax), random.randint(ymin, ymax)
        return i, j, Window(row_off=i, col_off=j, width=size, height=size)

    def _get_rasters(self, avg_density: float, size: int, no_data_percentage: float, output_path: str):
        """Performs the iteration, calls the get_window function and save the rasters.

        Args:
            avg_density (float): Average density of each pixel to finish algorithm.
            size (int): Square size of the output images
            no_data_percentage (float): Percentage used to filer images. If an image has more nodata than this value it will be discarded
            output_path (str): Where to generate the now geofiles
        """
        self.distribution_array = np.zeros_like(self.array)
        self.counter_array = np.zeros_like(self.array)

        n_image = 0
        pbar = tqdm()
        while True:
            self.check_iteration()

            i, j, window = self.get_window(size)
            new_array, profile, valid = self.get_raster_from_window(window, size, no_data_percentage)

            self.counter_array[i: size + i, j: size + j] += 1
            mean_density = round(self.counter_array.mean(), 2)
            pbar.set_description(f"Mean avg density {mean_density}")
            if not valid:
                continue

            self.distribution_array[i: size + i, j: size + j] += 1
            self.save_raster(new_array, output_path, profile, valid, f"{self.identifier}_randchete_{i}")

            n_image += 1
            pbar.update(1)
            if mean_density >= avg_density:
                break
