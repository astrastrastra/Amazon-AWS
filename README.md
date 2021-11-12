# Intelligent Measurement Systems


### The purpose of the project 
Design and implement a measurement system that analyzes data on terrain height differentiation by selecting groups of areas with the highest growth (Europe continent). The height increase in a given location should be measured with at least 10 measurement points. Set 5 groups of areas by the average value of the height increase. Mark the detected areas on the map. 

## Overview
Our first idea was to create a simple system based on the Python's PODPAC library, which simplifies the process of extracting portions of data that are in the interest zone. Taking the geographic coordinate system into considerations, we scrapped the desired lattitude and longitude using Google Maps. Our results are available in the *ISP_v1.py* file, although we abandoned this idea, as with the increase of data amount it was crucial to use a distributed system capable of parallelizing the computations. Higher zooms taken from the dataset caused memory errors.

With the desire to increase robustness and stability of the system, we decided to use the provided AWS services. We concentrated on setting up an Elastic Map Reduce environment
on the EC2 instance with Jupyter Notebook support, so that the whole process would be easier to manipulate with.

## Code

Listed below there are the code sections with a brief description of each block.

All the necessary imports:

    import rasterio
    from rasterio.io import MemoryFile
    import numpy as np
    import pyspark
    from pyspark.sql import SparkSession
    from pyspark import SparkContext
    from matplotlib import pyplot as plt

Initialize Spark session objects:

    sc = SparkContext()
    spark = SparkSession(sc)

Set up map tiles coordinates and zoom, corresponding to the Europe's area:

    zoom = 6
    coords_top_left = [14*(2**(zoom-5)), 7*(2**(zoom-5))]
    if zoom>5:
    	coords_bottom_right = [19*(2**(zoom-5))+1, 12*(2**(zoom-5))+1] 	
    else:
    	coords_bottom_right = [19*(2**(zoom-5)), 12*(2**(zoom-5))]
    
Calculate the width and height of the area - map tiles count:

    tiles_x = abs(coords_top_left[0]-coords_bottom_right[0])
    tiles_y = abs(coords_top_left[1]-coords_bottom_right[1])
    
Generate an array, storing paths of images representing tiles in the ROI:

    def generate_zoom_tiles():
        zoom_tiles = []
    
        for x in range(coords_top_left[0],coords_bottom_right[0]):
            for y in range(coords_top_left[1],coords_bottom_right[1]):
                zoom_tiles.append("s3://elevation-tiles-prod/geotiff/"+str(zoom)+"/"f"{x}/{y}.tif")
        return zoom_tiles
 
Handle the sequence of data bytes using rasterio's *MemoryFile* class:

    def unravel_data(byte):
        with MemoryFile(byte) as memfile:
            with memfile.open() as dt:
                data_arr = dt.read()
                return data_arr
                
The aim of the project was to process the land data only, so all of the negative altitude values (below sea level) are zeroed out:

    def remove_water(data_dist_array):
        np.place(data_dist_array, data_dist_array<0, 0)
        return data_dist_array
        
Altitude data processing using sub-tiled data - all of the tiles are remodeled into sub-tiles of given shape. Every sub-tile is then processed by the elevation extracting algorithm resulting in an output data shape reduction. The elevation value is a mean absolute bidirectional gradient of the altitude matrices:

    def get_mean_value(data_dist_array):
    val = data_dist_array[0]
	
    TILE_SIZE=512
    SUB_TILE_SIZE=4*(2**(zoom-5))
	
    if val.shape[0] != TILE_SIZE:
        SUB_TILE_SIZE = int(SUB_TILE_SIZE // (TILE_SIZE//val.shape[0]))
		
    tile = val.reshape(TILE_SIZE//SUB_TILE_SIZE, SUB_TILE_SIZE, -1, SUB_TILE_SIZE).swapaxes(1, 2).reshape(-1, SUB_TILE_SIZE, SUB_TILE_SIZE)
    deltas = np.mean(np.mean(np.abs(np.gradient(tile, axis=(1,2))), axis=0), axis=(1,2))
	
    return deltas.reshape((val.shape[0]//SUB_TILE_SIZE, val.shape[0]//SUB_TILE_SIZE))
        
Converting elevation data into classes specified by the bin ranges array:

    def digitize_tiles(data_dist_array, bins):
        return np.digitize(data_dist_array, bins)

Minimum and maximum array value extraction:

    def get_min_val(data_dist_array):
        return np.min(data_dist_array)
        
    
    def get_max_val(data_dist_array):
        return np.max(data_dist_array)
            
Get image paths and load the data into the RDD as binary files:

    zoom_tiles = generate_zoom_tiles()

    data_dist = sc.binaryFiles(",".join(zoom_tiles))

Extract and convert the content from the RDD (path, data) tuples to simplify further processing:

    data_dist_values = data_dist.map(lambda x: bytes(x[1]))

Unravel, preprocess and extract the elevation data:

    data_dist_array = data_dist_values.map(lambda x: unravel_data(x))
    data_dist_nowater = data_dist_array.map(lambda x: remove_water(x))
    data_dist_deltas = data_dist_nowater.map(lambda x: get_mean_value(x))

Extract the boundary values and calculate the bin sizes spaced in a geometric progression sequence:

    data_dist_min = min(data_dist_deltas.map(lambda x: get_min_val(x)).collect())
    data_dist_max = max(data_dist_deltas.map(lambda x: get_max_val(x)).collect())

    bins = np.geomspace(10, data_dist_max, 6)

Convert elevation data into classes corresponding to calculated bins:

    data_dist_digitized = data_dist_deltas.map(lambda x: digitize_tiles(x, bins))
    
Collect data from the RDD and stack the output tiles to form a single map image:

    tiles_digitized = data_dist_digitized.collect()
    map_rows=[]
    TILE_NUM = 20


    for y in range(tiles_x):
        map_rows.append(np.vstack(tiles_digitized[y*tiles_y:y*tiles_y+tiles_y]))

    map_all = np.hstack(map_rows)

## Setup

We’ve created two approaches, one using **PuTTY** and **Hadoop**, and a second one with **Jupyter Notebook**. To run presented code one needs to start with creating their own key pair. In order to do that, the steps below are required: 

**Services** -> **EC2** -> **EC2 Dashboard** -> **Resources window** -> **Key pairs** -> **Create key pair** -> **ppk** -> **Create key pair** 

Key pair downloads itself instantly on a local machine. Next step is to run a desired cluster. One needs to go successively to:

**Services** -> **EMR** -> **Clusters** -> **Create cluster** -> **Advanced options** -> **Spark** (apart from default)

If desired to run through a Jupyter Notebook, also JupyterHub, JupyterEnterpriseGateway and Livy are mandatory.

For default 1 Master and 2 Core we used m4.xlarge instantion on each node as it supported Jupyter Notebook. The choice is arbitrary for an user though, although we can’t guarantee that Jupyter Notebook nor every of the instances will be available. The remaining steps are to set a personal cluster name and attach previously created EC2 key pair from a list. Since our code is running several libraries that are not genuinely attached in the cluster, one needs to manually install them on every instance, following given pipeline.

First, it is obligatory to add additional inboud rules in: 

**Services** -> **EC2** -> **Security Groups**.

In each of the listed groups, one needs to:

**Edit inbound rules** -> **Add rule** -> **SSH & Anywhere** -> **Save rules**

After that, next up in:

**Services** -> **EC2** -> **Instances**

Each of the active nodes has its own **Public IPv4 DNS** which will be used used to connect to it via *ssh*. On the local machine, open PuTTY:

**Host name** -> **hadoop@< Public IPv4 DNS >**

And then go to:

**Connection** -> **ssh** -> **Auth** -> **Private key file for authentication** 

and add previously created .ppk file that contains the key pair. Establish the connection and write down the following command:

    sudo pip3 install matplotlib rasterio
    
Afer  repeating the process on each node, one is good to go to run the code. 

### Option: Hadoop & PuTTY

Transfer the file <ISP.py> using i.e. WinSCP to a master node. Then simply run by typing:

    spark-submit <ISP.py>

### Option: Jupyter Notebook

In order to use the code in Jupyter notebook it is additionally needed to navigate to:

**Services** -> **EMR** -> **Notebooks** -> **Create notebook**

and there choose a previously set up cluster and apply. Then simply go to Notebooks again, select a desired one and use the *Open in JupterLab* box. The remaining step, providing it will not be automatically selected, is to pick a *PySpark* kernel at the top-right of the page.

## Results

<center>
	
| Zoom		| Time (EMR) [s]| Time(local) [s] |
| ------------- |:-------------:|:---------------:|
| 5     	| 15.46	        | 22.42	          |
| 6     	| 34.96         | 73.21           |
| 7 		| 162.54        | 277.76          |
| 8     	| 465.54        |      	-	  |
| 9 		| 1029.45       |      	-	  |
| 10     	| 2906.65       |      	-	  |
| 11		| 11103.55	|	-	  |

</center>

Using the EMR and EC2 cluster (m4.xlarge, 1 master, 2 slaves configuration), we were able to process higher amount of data. The goal was to divide an increase of terrain altitude in Europe into 6 groups. We decided that apart from a linear scale, also a logarithmic one will be used in assigning the groups, in order to increase the visual diversity of results. We've run several benchmarks of our system in order to test how it performs. In addition to that, we also included results of our local machine performance (equipped with Intel Xeon E5-1650 and 16GB of RAM), although we were unable to process data above zoom level 7.
The table above yields relation between precision of calculations and the amount of time it took to sucessfully compute the averages. In addition to the table, we decided to visualize the results on a chart below. It is clearly visible, that computation time is exponentially bounded with zoom growth.

<p align="center">
  <img width="726" height="497" src="https://i.gyazo.com/9df532eac705898510f48b9cf90cf1ad.png">
</p>

Charts covering the average altitude increase have been placed in the *results* folder.
