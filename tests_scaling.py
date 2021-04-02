#!/usr/bin/env python3

import numpy as np
from sklearn.preprocessing import minmax_scale

import processing

a = np.array([[1, 0, 2],
              [1, 5, 2],
              [1, 0, 0]])

b = np.array([[-1, 1, 3],
              [2, 0, 2],
              [3, 1, 1]])

print(processing.scale_minmax(b))
print(processing.scale_audio(b))
