import os
import re
from pathlib import Path

import rasterio as rio
from rasterio.merge import merge


class MergeTiffs(): 
    
    def __init__(self, folder, output_folder="") -> None:
        self.folder = folder
        self.output_folder = os.path.join(folder, "merged") if not output_folder else output_folder
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        
        
    def process(self, output_filename = "mosaic_output.tif"): 
        
        w_path = Path(self.folder)
        output_filename = os.path.join(self.output_folder, output_filename)
        
        raster_files = [p for p in w_path.iterdir() if re.search(".tiff?$", p.__str__())]
        raster_to_mosiac = []
        for p in raster_files:
            raster = rio.open(p)
            raster_to_mosiac.append(raster)
            
        mosaic, output = merge(raster_to_mosiac)
        
        output_meta = raster.meta.copy()
        output_meta.update(
            {"driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": output,
            }
        )
        
        with rio.open(output_filename, "w", **output_meta) as m:
            m.write(mosaic)