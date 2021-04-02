
import PIL.Image
import numpy as np

PIL.Image.MAX_IMAGE_PIXELS = 240000000

fpath = '/media/findux/DATA/R/data/ETOPO1_Ice_g_geotiff.tif'
data = np.array(PIL.Image.open(fpath))
print(data.shape)