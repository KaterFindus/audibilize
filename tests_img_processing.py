#!/usr/bin/env python3

import cv2
import matplotlib.pyplot as plt
import numpy as np


fpath = '/media/findux/DATA/R/data/ETOPO1_Ice_g_geotiff.tif'
res = 0.05

img = cv2.imread(fpath)

height, width = img.shape[:2]

h = int(height * res)
w = int(width * res)


img_small = cv2.resize(img, (w, h))
cv2.imshow('Resized', img_small)
while True:
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break