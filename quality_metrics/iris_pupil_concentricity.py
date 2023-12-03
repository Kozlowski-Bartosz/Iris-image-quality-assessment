import cv2
import numpy as np
import math
from collections import namedtuple

class IrisPupilCocentricityCalc:
    
    def __init__(self, img, pupil_coords, iris_coords):
        self.img = img
        self.pupil = namedtuple('pupil', ['x', 'y', 'r'])(pupil_coords[0], pupil_coords[1], pupil_coords[2])
        self.iris = namedtuple('iris', ['x', 'y', 'r'])(iris_coords[0], iris_coords[1], iris_coords[2])
    
    def calculate_cocentricity(self):
        distance = (math.sqrt(pow((self.pupil.x - self.iris.x),2) + pow((self.pupil.y - self.iris.y),2)))/(self.iris.r)
        return 100 * max(1-distance, 0)