#!/usr/bin/env python3

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import write
from PIL import Image
import time

import processing

Image.MAX_IMAGE_PIXELS = 240000000

sampling_rate = 44100
duration = 0.25
Hz_start = 800
Hz_stop = 6000
Hz_res = 25
y_res = 500

fpath_img = '/media/findux/DATA/R/data/ETOPO1_Ice_g_geotiff.tif'
# fpath_img = '/media/findux/DATA/Code/audibilize/test_img.tiff'

fname_out = '/home/findux/Desktop/signal_output.wav'
fname_checkpoint_aligned = '/home/findux/Desktop/aligned.npy'

hz_vec = np.linspace(Hz_start, Hz_stop, int((Hz_stop - Hz_start) / Hz_res))
hz_vec = np.around(hz_vec, 2)
print(f'Hz vector length: {len(hz_vec)}')

print('Loading data.')
data = np.array(Image.open(fpath_img))
if data.ndim > 2:
    data = data[:, :, 0]
print(data.shape)

print('Y binning.')
y_bins = np.array_split(data, y_res, axis=0)
y_binned = np.zeros((1, data.shape[1]))
for horz_batch in y_bins:
    bin_avg = horz_batch.mean(axis=0)
    y_binned = np.vstack((y_binned, bin_avg))
data = y_binned[1:]
print(data.shape)

# Align data
print('Aligning data with Hz bins.')
data_aligned = processing.align_data(data, hz_vec, verbose=True)
print('aligned:', type(data_aligned), data_aligned.shape)
print('Saving checkpoint.')
np.save(fname_checkpoint_aligned, data_aligned)

# # Display data
# processing.array_image(data_aligned)

# Scale data
print('Scaling data.')
data_scaled = processing.scale_minmax(data_aligned)

# Extract signals
print('Extracting signals.')
signals = processing.signals_from_array(data_scaled, hz_vec, duration_s=duration, verbose=True)

# Glue signals together
signals_flat = signals.flatten()
# signals_flat = processing.scale_audio(signals_flat)

timestamp = time.time()
figname = '/home/findux/Desktop/signal_plot_' + str(timestamp) + '.png'
plt.plot(range(len(signals_flat)), signals_flat)
plt.show()
plt.savefig(figname)

# Save raw signal
print('Saving raw signal')
np.save('/home/findux/Desktop/raw_signal.npy', signals_flat)

# Save signal
write(fname_out, sampling_rate, signals_flat)