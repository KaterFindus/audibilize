#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import write

fpath_out = '/home/findux/Desktop/test.wav'
sampling_rate = 44100

t = np.linspace(0, 2, sampling_rate)
signal = np.sin(t * np.pi * 440)

plt.plot(t, signal)
plt.show()

write(fpath_out, sampling_rate, signal)