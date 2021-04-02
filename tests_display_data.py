import numpy as np
from PIL import Image

fname_checkpoint_aligned = '/home/findux/Desktop/aligned.npy'

data_aligned = np.load(fname_checkpoint_aligned)

img = Image.fromarray(data_aligned)
img.show()
