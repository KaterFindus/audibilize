#!/usr/bin/env python3

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wf
import sounddevice as sd

fpath_DEM = '/media/findux/DATA/R/data/ETOPO1_Ice_g_geotiff.tif'
out = '/home/findux/Desktop/test.wav'

data = cv2.imread(fpath_DEM)
data = data[:, :, 0]
print(data.shape, type(data))

slice_count, bin_count = data.shape

xmin = 20
xmax = 20000
xvals = np.linspace(xmin, xmax, bin_count)
assert len(xvals) == bin_count

# # Plot one line:
# plt.bar(xvals, data[1500])
# plt.xlabel('Hz')
# plt.ylabel('Intensity\n(not normalized)')
# plt.show()

sd.play(data, samplerate=44100)


# data_scaled = np.interp(data, (np.min(data), np.max(data)), (-1, 1))
# wf.write(out, 44100, data_scaled)