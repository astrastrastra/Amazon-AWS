from podpac.datalib.terraintiles import TerrainTiles
from podpac import Coordinates, clinspace
import numpy as np
from matplotlib import pyplot as plt
import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import StringType, IntegerType,StructType,FloatType,StructField


spark = SparkSession.builder.appName("TerrainTiles").getOrCreate()
sc = spark.sparkContext

TILE_NUM=10000
TILE_SIZE=25
if TILE_NUM%TILE_SIZE!=0:
    exit(-1)

# Get tile data
node = TerrainTiles(tile_format='geotiff', zoom=5)
coords = Coordinates([clinspace(71, 35, TILE_NUM), clinspace(-25, 39, TILE_NUM)], dims=['lat', 'lon'])
ev = node.eval(coords)
mapa = np.asarray(ev.data)
mapa[mapa<0] = 0
map_tiled = mapa.reshape(TILE_NUM//TILE_SIZE, TILE_SIZE, -1, TILE_SIZE).swapaxes(1, 2).reshape(-1, TILE_SIZE, TILE_SIZE)
distData = sc.parallelize(list(map_tiled))

# Elevation param extraction
lows = np.amin(map_tiled, axis=(1, 2))
highs = np.amax(map_tiled, axis=(1, 2))
deltas = highs-lows

# Prep bins
delta_min = min(deltas)
delta_max = max(deltas)
# bins = np.linspace(delta_min, delta_max, 10)
bins = np.geomspace(10, delta_max, 5)

# Categorize as bins and rescale to match input
deltas_digitized = np.digitize(deltas, bins)
deltas = deltas.reshape((TILE_NUM//TILE_SIZE, -1))
deltas_org = np.kron(deltas, np.ones((TILE_SIZE, TILE_SIZE), dtype=int))

deltas_digitized = deltas_digitized.reshape((TILE_NUM//TILE_SIZE, -1))
plt.imsave("wynik.png", deltas_digitized)
plt.imsave("org_shape.png", deltas_org)
