#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf

dt = 0.001  # delta t: 1 kHz (sampling rate)
t = np.arange(0, 2, dt)  # 2 seconds of audio w/ the specified dt
f0 = 50
f1 = 250
t1 = 2

x = np.cos(2 * np.pi * t * (f0 + (f1 - f0) * np.power(t, 2) / (3 * t1 ** 2)))

fs = 1 / dt

# sd.play(2 * x, fs)
audio = sd.Stream.write(x)