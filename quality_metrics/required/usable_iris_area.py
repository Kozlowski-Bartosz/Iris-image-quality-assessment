import cv2
import numpy as np
from helper_functions import convert_OSIRIS_coords_to_xyr

class UsableIrisAreaCalc:
    
    THRESH = 70.0
    
    def __init__(self, img, iris_coords, pupil_coords, segmentation_mask):
        self.img = img
        self.iris_coords = iris_coords
        self.pupil_coords = pupil_coords
        self.segmentation_mask = segmentation_mask

    def generate_iris_mask(self):
        mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        pupil_x, pupil_y, pupil_radius = convert_OSIRIS_coords_to_xyr(self.pupil_coords)
        iris_x, iris_y, iris_radius = convert_OSIRIS_coords_to_xyr(self.iris_coords)
        cv2.circle(mask, (int(iris_x), int(iris_y)), int(iris_radius), 255, -1)  # Draw the iris
        cv2.circle(mask, (int(pupil_x), int(pupil_y)), int(pupil_radius), 0, -1)  # Exclude the pupil
        return mask

    def count_masked_pixels(self, mask):
        return np.count_nonzero(self.img[mask == 255])

    def create_occlusion_mask(self):
        iris_mask = self.generate_iris_mask()
        return iris_mask - self.segmentation_mask

    def calculate_usable_area(self):
        whole_mask = self.generate_iris_mask()
        occlusion_mask = self.create_occlusion_mask()
        cv2.imshow("Whole mask", whole_mask)
        cv2.imshow("Occlusion mask", occlusion_mask)
        iris_pixels = self.count_masked_pixels(whole_mask)
        occluded_pixels = self.count_masked_pixels(occlusion_mask)
        return (1 - occluded_pixels / iris_pixels) * 100
    
    def assert_area_above_thresh(self, area):
        return area >= self.THRESH
