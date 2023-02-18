import random

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

    def _get_rasters(self, n_images: int, size: int, no_data_percentage: float, output_path: str):
        """Performs the iteration, calls the get_window function and save the rasters.

        Args:
            n_images (int): Output number of images
            size (int): Square size of the output images
            no_data_percentage (float): Percentage used to filer images. If an image has more nodata than this value it will be discarded
            output_path (str): Where to generate the now geofiles
        """
        self.final_counter_array = np.zeros_like(self.array)
        self.counter_array = np.zeros_like(self.array)

        n_image = 0
        pbar = tqdm(total=n_images)
        while True:
            self.check_iteration()

            i, j, window = self.get_window(size)
            new_array, profile, valid = self.get_raster_from_window(window, size, no_data_percentage)

            self.counter_array[i: size + i, j: size + j] += 1
            if not valid:
                continue

            self.final_counter_array[i: size + i, j: size + j] += 1
            self.save_raster(new_array, output_path, profile, valid, f"{self.identifier}_randchete_{i}")

            n_image += 1
            pbar.update(1)
            if n_image >= n_images:
                break
