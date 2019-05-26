import cv2
import numpy as np 
from pathlib import Path
import math
import matplotlib.pyplot as plt
import scipy.ndimage 

#contour = partition of the contour that contains the ridges

class PointDetection:

    def __init__(self, contour):
        self.contour = contour
        self.left = []
        self.right = []

def find_gaussian(self, contour, sigma=4):
    gaussian = scipy.ndimage.gaussian_filter(contour, sigma)

    return gaussian

def find_key_points(self, gaussian):
    prev = 0
    current = 0

    for point in np.diff(gaussian):
        if prev < 0 <= point:
            self.left.append(current)
   







