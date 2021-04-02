#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

Hz = 25
x = np.linspace(0, 2, 44100)
y = np.sin(x * np.pi * Hz)

plt.plot(x,y)
plt.show()