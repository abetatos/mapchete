import random

import numpy as np
from tqdm import tqdm
from rasterio.windows import Window

from .basechete import BaseChete


class MaxChete(BaseChete):

    def get_window(self, size, no_data_percentage):
        output_length = size**2

        stride = int(max(size/10, random.random()*(size-1)))

        x_iterations = int(np.floor(abs(self.height-size)/stride))
        y_iterations = int(np.floor(abs(self.width-size)/stride))

        tmp_aux = np.zeros((x_iterations, y_iterations))
        for i in range(x_iterations):
            i_init = i*stride
            i_end = i*stride + size
            for j in range(y_iterations):
                j_init = j*stride
                j_end = j*stride + size
                if j == 0:
                    tmp = self.counter_array[i_init: i_end, j_init: j_end]
                    tmp_aux[i, j] = tmp.sum()
                else:  # Just for optimization purposes. Reduction of ~20% in execution time
                    old_value = tmp_aux[i, j-1]
                    delta = (
                        - self.counter_array[i_init: i_end, (j-1) * stride: j_init].sum()
                        + self.counter_array[i_init: i_end, (j-1) * stride + size: j_end].sum()
                    )
                    tmp_aux[i, j] = old_value + delta

        minimums = np.argwhere(tmp_aux == tmp_aux.min())
        index = int(random.random() * minimums.shape[0])
        min_i, min_j = minimums[index]*stride

        output_array = self.array[min_i: size + min_i, min_j: size + min_j] 
        no_data_count = output_array[output_array == self.nodata].shape[0]

        # Update density
        if no_data_count/output_length > no_data_percentage:
            mask = (output_array == self.nodata)
            self.counter_array[min_i: size + min_i, min_j: size + min_j][mask] += 1 
            return None
        else:
            self.counter_array[min_i: size + min_i, min_j: size + min_j] += 1
            self.distribution_array[min_i: size + min_i, min_j: size + min_j] += 1
            return Window(min_j, min_i, size, size)

    def _get_rasters(self, n_images: int, size: int, no_data_percentage: float, identifier: str, output_path: str):
        self.counter_array = np.zeros_like(self.array)
        self.distribution_array = np.zeros_like(self.array)

        n_image = 0
        pbar = tqdm(total=n_images)
        while True:
            self.check_iteration()

            window = self.get_window(size, no_data_percentage)
            if window is None:
                self.valid_dict[0] += 1
                continue

            new_array, profile, valid = self.get_raster_from_window(window, size, no_data_percentage)
            self.save_raster(new_array, output_path, profile, valid, f"{identifier}_maxchete_{n_image}")

            n_image += 1
            pbar.update(1)
            if n_image >= n_images:
                break
