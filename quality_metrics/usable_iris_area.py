import cv2
import numpy as np
from helper_functions import generate_empty_mask
from collections import namedtuple

class UsableIrisAreaCalc:
    
    THRESH = 70.0

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

    def create_occlusion_mask(self):
        iris_mask = self.generate_iris_mask()
        return iris_mask - self.segmentation_mask

    def calculate_usable_area(self):
        whole_mask = self.generate_iris_mask()
        occlusion_mask = self.create_occlusion_mask()
        iris_pixels = self.count_masked_pixels(whole_mask)
        occluded_pixels = self.count_masked_pixels(occlusion_mask)
        return (1 - occluded_pixels / iris_pixels) * 100
    
    def assert_area_above_thresh(self, area):
        return area >= self.THRESH
