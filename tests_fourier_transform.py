#!/usr/bin/env python3

import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.io.wavfile as wf
import sounddevice as sd

random.seed(42)
fpath_DEM = '/media/findux/DATA/R/data/ETOPO1_Ice_g_geotiff.tif'
n_subsample = 1

data = cv2.imread(fpath_DEM)
data = data[:, :, 0]
print(data.shape, type(data))

slice_count, bin_count = data.shape
rand_i = random.sample(list(range(slice_count)), n_subsample)  # does not feature replacement

xmin = 20
xmax = 20000

xvals = np.linspace(xmin, xmax, bin_count)
assert len(xvals) == bin_count


subsample = data[rand_i]
# for spec in data[rand_i, :]:
#     plt.plot(xvals, spec)
# plt.grid()
# plt.show()

# Get the fourier transform of the signal
signal_ftf = np.fft.fft(subsample)[0]
print(signal_ftf.shape)
# only use half the signal (single-sided spectrum)
signal_ftf = signal_ftf[:int(bin_count * 0.5)]
# Divide the result by the signal length
signal_ftf = signal_ftf / bin_count
# Get rid of th eimaginary numbers
signal_ftf = abs(signal_ftf)

# plt.plot(signal_ftf)
# plt.show()

