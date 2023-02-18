import numpy as np
from rasterio.windows import Window
from tqdm.auto import tqdm

from .basechete import BaseChete


class SeqChete(BaseChete):

    def get_window(self, size: int, i: int, j: int):
        """It generates windows based on a probabilistic approach. It generates a map sequentially filtering the images
        that have a higher percentage of nodata than the no_data_percentage value. 

        Args:
            size (int): Output size of tiles
            i (int): row offset
            j (int): column offset

        Returns:
            rasterio.windows.Window: A rasterio window object
        """
        return Window(row_off=i, col_off=j, width=size, height=size)

    def _get_rasters(self, n_images: int, size: int, no_data_percentage: float, output_path: str):
        """Performs the iteration, calls the get_window function and save the rasters.

        Args:
            n_images (int): Output number of images
            size (int): Square size of the output images
            no_data_percentage (float): Percentage used to filer images. If an image has more nodata than this value it will be discarded
            output_path (str): Where to generate the now geofiles
        """
        self.distribution_array = np.zeros_like(self.array)
        self.counter_array = np.zeros_like(self.array)

        images_done = 0
        i_steps = int(np.ceil(self.height/size))
        j_steps = int(np.ceil(self.width/size))

        pbar = tqdm(total=n_images)
        while True:
            self.check_iteration()
            for i in range(i_steps):
                for j in range(j_steps):
                    window = self.get_window(size, i*size, j*size)
                    new_array, profile, valid = self.get_raster_from_window(window, size, no_data_percentage)

                    self.counter_array[i: size + i, j: size + j] += 1
                    if not valid:
                        continue
                    self.distribution_array[i*size: (i+1)*size, j*size: (j+1) * size] += 1
                    self.save_raster(new_array, output_path, profile, valid, f"{self.identifier}_seqchete_{images_done}")

                    images_done += 1
                    pbar.update(1)
                    if images_done >= n_images:
                        break
                else:
                    continue
                break
            else:
                continue
            break
