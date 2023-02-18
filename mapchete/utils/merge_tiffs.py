import os
import re
from pathlib import Path

import logging
import rasterio as rio
import rasterio.plot as rplt
from rasterio.merge import merge

logger = logging.getLogger("Clip")
logger.setLevel(logging.INFO)


def merge_tiffs(folder: str = "raster_clip", output_folder: str = None, output_filename = "mosaic_output.tif"):

    output_folder = os.path.join(folder, "merged") if not output_folder else output_folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    w_path = Path(folder)
    output_filename = os.path.join(output_folder, output_filename)

    raster_files = [p for p in w_path.iterdir() if re.search(".tiff?$", p.__str__())]
    raster_to_mosiac = []
    for p in raster_files:
        raster = rio.open(p)
        raster_to_mosiac.append(raster)

    if not raster_to_mosiac:
        raise FileNotFoundError("Looks like your directory does not have any tiff file")

    mosaic, output = merge(raster_to_mosiac)

    output_meta = raster.meta.copy()
    output_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": output,
    })

    with rio.open(output_filename, "w", **output_meta) as m:
        m.write(mosaic)

    rplt.show(mosaic, cmap="Greys")
    logger.info(f"File created in {output_filename}")
