import cv2
import numpy as np
from helper_functions import convert_OSIRIS_coords_to_xyr

class ContrastCalc:
    
    IRIS_SCLERA_THRESH = 5.0
    IRIS_PUPIL_THRESH = 30.0
    
    def __init__(self, img, iris_coords, pupil_coords):
        self.img = img
        self.iris_coords = iris_coords
        self.pupil_coords = pupil_coords
    
    def generate_area_mask(self, inner_radius, outer_radius, mask_name = "Area mask", display_mask = True):
        mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        iris_x, iris_y, iris_radius = convert_OSIRIS_coords_to_xyr(self.iris_coords)
        
        cv2.ellipse(mask, (int(iris_x), int(iris_y)), (int(outer_radius), int(outer_radius)), 0, -30, 30, 255, -1)
        cv2.ellipse(mask, (int(iris_x), int(iris_y)), (int(outer_radius), int(outer_radius)), 180, -30, 30, 255, -1)    
        
        cv2.ellipse(mask, (int(iris_x), int(iris_y)), (int(inner_radius), int(inner_radius)), 0, -35, 35, 0, -1)
        cv2.ellipse(mask, (int(iris_x), int(iris_y)), (int(inner_radius), int(inner_radius)), 180, -35, 35, 0, -1)   
        
        cv2.imshow(mask_name, cv2.bitwise_and(self.img, self.img, mask=mask)) if display_mask else None
        
        return mask
          
    def generate_iris_area_mask(self):
        mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        pupil_x, pupil_y, pupil_radius = convert_OSIRIS_coords_to_xyr(self.pupil_coords)
        iris_x, iris_y, iris_radius = convert_OSIRIS_coords_to_xyr(self.iris_coords)
        
        inner_radius = (pupil_radius + iris_radius) / 2
        outer_radius = 0.9 * iris_radius
        
        return self.generate_area_mask(inner_radius, outer_radius, "Iris area mask")
    
    def generate_iris_area_mask_for_pupil(self):
        mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        pupil_x, pupil_y, pupil_radius = convert_OSIRIS_coords_to_xyr(self.pupil_coords)
        iris_x, iris_y, iris_radius = convert_OSIRIS_coords_to_xyr(self.iris_coords)
        
        inner_radius = 1.1 * pupil_radius
        outer_radius = (pupil_radius + iris_radius) / 2
        
        return self.generate_area_mask(inner_radius, outer_radius, "Iris area mask for pupil")
    
    def generate_sclera_area_mask(self):
        mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        iris_x, iris_y, iris_radius = convert_OSIRIS_coords_to_xyr(self.iris_coords)
        
        inner_radius = 1.1 * iris_radius
        outer_radius = 1.2 * iris_radius
        
        return self.generate_area_mask(inner_radius, outer_radius, "Sclera area mask")
    
    def generate_pupil_area_mask(self):
        mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        pupil_x, pupil_y, pupil_radius = convert_OSIRIS_coords_to_xyr(self.pupil_coords)
        
        calculated_radius = 0.8 * pupil_radius
        
        cv2.circle(mask, (int(pupil_x), int(pupil_y)), int(calculated_radius), 255, -1)
        
        cv2.imshow("Pupil area mask", cv2.bitwise_and(self.img, self.img, mask=mask))
        
        return mask
        
        
        
    def get_iris_value(self):
        iris_mask = self.generate_iris_area_mask()
        return np.median(self.img[iris_mask == 255])
    
    def get_iris_value_for_pupil(self):
        iris_mask = self.generate_iris_area_mask_for_pupil()
        return np.median(self.img[iris_mask == 255])
    
    def get_sclera_value(self):
        sclera_mask = self.generate_sclera_area_mask()
        return np.median(self.img[sclera_mask == 255])
    
    def get_pupil_value(self):
        pupil_mask = self.generate_pupil_area_mask()
        return np.median(self.img[pupil_mask == 255])
    
    
    
    def calculate_iris_sclera_contrast(self):
        iris_value = self.get_iris_value()
        sclera_value = self.get_sclera_value()
        pupil_value = self.get_pupil_value()
        
        if (pupil_value >= iris_value) or (pupil_value >= sclera_value):
            return 0
        else:
            return 100 * abs(sclera_value - iris_value) / (sclera_value + iris_value - (2 * pupil_value))
        
    def calculate_iris_pupil_contrast(self):
        pupil_value = self.get_pupil_value()
        iris_value = self.get_iris_value_for_pupil()
        
        weber_ratio = abs(iris_value - pupil_value)/(20 + pupil_value)
        
        return weber_ratio * 100 / (0.75 + weber_ratio)

    def assert_iris_sclera_contrast_above_thresh(self, contrast):
        return contrast >= self.IRIS_SCLERA_THRESH
    
    def assert_iris_pupil_contrast_above_thresh(self, contrast):
        return contrast >= self.IRIS_PUPIL_THRESH