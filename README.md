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

You can see the differences between both images, and that effectively, matchete can optimize the process. 

Random | MAXchete
--- | ---
![](https://user-images.githubusercontent.com/76526314/219666167-64e7f0a8-df76-4422-8665-a6f908b0a98b.png) | ![](https://user-images.githubusercontent.com/76526314/219665645-7eefad2e-bc33-43cb-99fa-5374f6c84ea4.png)

The input file was a geotiff of shape: 


# How does it work?

There are three ways of creating your dataset: 

- random
- sequential
- maxchete

Just instantiate the class and machete the data!

```python 
mchete = MAPchete(input_file, size=512, output_path="raster_clip", clear_output_path=True)
mchete.get_rasters("maxchete", identifier=input_file, n_images=100, no_data_percentage = 0.2)

```

