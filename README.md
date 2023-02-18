<p align="center">
  <img width="300" alt="mapchete_final" src="https://user-images.githubusercontent.com/76526314/219464092-ee4e075c-c8c7-4d39-8017-cb0ede17248f.png">
</p>

<h3 align="center">
    <p>Cut your geospatial data into smaller pieces</p>
</h3>

# MAPchete

Welcome to my Github project! This repository was created to assist with the preparation of geospatial data for deep learning purposes. Specifically, the project focuses on efficiently cropping large datasets into smaller tiles, with the goal of generating a dataset with minimum overlap between tiles (Which could result in the loss of representativeness). This process is essential for achieving optimal model performance, and can be applied to various other applications within the geospatial imagery field. Thank you for checking out my project, and feel free to explore the code and contribute to its development!

# What does MAPchete has to offer?

It generates patches based on a probabilistic approach that tries to augment the covered area distributing images more efficiently while avoiding images with a nodata percentage avobe a given threshold. It is perfect for deep learning purposes as it will maximize the outcome of your model! 

If we generate the dataset randomly we can see that there are zones that have great number of tiles, while with this approach you can obtain a more well distributed dataset. 

With an input of shape: 
<p align="center">
  <img width="300" alt="mapchete_final" src="https://user-images.githubusercontent.com/76526314/219682129-756f265c-6f4c-4c20-bc2e-bc5e438f4721.png">
</p>

Mapchete obtains: 

Iterations | RANDchete (random) | MAXchete (maximize)
--- | --- | ---
100 iter | ![](https://user-images.githubusercontent.com/76526314/219666167-64e7f0a8-df76-4422-8665-a6f908b0a98b.png) | ![](https://user-images.githubusercontent.com/76526314/219665645-7eefad2e-bc33-43cb-99fa-5374f6c84ea4.png)
1000 iter | ![image](https://user-images.githubusercontent.com/76526314/219706410-985e57b5-5698-49e6-afdb-856fe01c073b.png) | ![image](https://user-images.githubusercontent.com/76526314/219707072-d8134441-64ba-41a3-a23a-74466f6c5bda.png)
std (1000 iter) | &plusmn; 9.1  |  &plusmn; 3.9


Compared to the random sample, the data is now more evenly distributed and the standard deviation has been reduced by over 2 times which shows library's effectiveness.

# How does it work?

There are three ways of creating your dataset: 

- randchete -> Random approach 
- seqchetel -> Sequential approach 
- maxchete -> Distribution approach

Just instantiate the class and machete the data!

```python 
from mapchete import FARMchete

maxchete = FARMchete(input_file).get("maxchete")
maxchete.get_rasters(n_images=100, size=512 , no_data_percentage = 0.3, output_path="raster_clip", clear_output_path=True)
maxchete.plot_bands()
```

<p align="center">
  <img width="400" alt="mapchete_final" src="https://user-images.githubusercontent.com/76526314/219875276-3a05f852-d68b-4f41-a684-f48147edbda5.png">
</p>

### Run to get the tiles: 

```python
maxchete.get_rasters()
```

### Study the output
```python 
fig, ax = maxchete.get_3Ddistribution()
```

<p align="center">
  <img width="500" alt="mapchete_final" src="https://user-images.githubusercontent.com/76526314/219876116-ce051ecf-021d-4996-bc1b-e68274f624b1.png">
</p>


There is another useful function called merge_tiffs which can merge generated images to se how the distribution was generated. If you use lower sampling, this becomes a useful tool, but if you opt for higher sampling, the algorithm should be capable of generating the complete extent of the original image.

``` python
from mapchete import merge_tiffs
merge_tiffs()
```
<p align="center">
  <img width="400" alt="mapchete_final" src="https://user-images.githubusercontent.com/76526314/219876203-2e36d9b6-9edf-4982-b9ba-c3d8c559c962.png">
</p>

# Instalation

At the moment there is no suitable distribution in PyPi, so installing must be done trough the setup.py
```
git clone https://github.com/abetatos/mapchete.git 
cd mapchete
python setup.py install
```

In older versions of python (<=3.7) rasterio gives some trouble installing, it is recommended to use newer versions (>3.7) or conda for the instalation. Rasterio version is not fixed in the requirements to give more versatility to the user but it was tested with version==1.2.10. 
