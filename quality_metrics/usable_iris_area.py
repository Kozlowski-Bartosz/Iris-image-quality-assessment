import cv2
import numpy as np
from helper_functions import generate_empty_mask
from collections import namedtuple

class UsableIrisAreaCalc:

    def __init__(self, img, iris_coords, pupil_coords, segmentation_mask):
        self.img = img
        self.iris = namedtuple('iris', ['x', 'y', 'r'])(iris_coords[0], iris_coords[1], iris_coords[2])
        self.pupil = namedtuple('pupil', ['x', 'y', 'r'])(pupil_coords[0], pupil_coords[1], pupil_coords[2])
        self.segmentation_mask = segmentation_mask

    def generate_iris_mask(self):
        mask = generate_empty_mask(self.img)
        cv2.circle(mask, (int(self.iris.x), int(self.iris.y)), int(self.iris.r), 255, -1)  # Draw the iris
        cv2.circle(mask, (int(self.pupil.x), int(self.pupil.y)), int(self.pupil.r), 0, -1)  # Exclude the pupil
        return mask

    def count_masked_pixels(self, mask):
        return np.count_nonzero(self.img[mask == 255])

    def calculate_usable_area(self):
        whole_mask = self.generate_iris_mask()
        usable_mask = self.segmentation_mask
        iris_pixels = self.count_masked_pixels(whole_mask)
        usable_pixels = self.count_masked_pixels(usable_mask)
        return (usable_pixels/iris_pixels) * 100