import cv2
import numpy as np
from helper_functions import generate_empty_mask
from collections import namedtuple

class ContrastCalc:
    
    IRIS_SCLERA_THRESH = 5.0
    IRIS_PUPIL_THRESH = 30.0
    
    def __init__(self, img, iris_coords, pupil_coords):
        self.img = img
        self.iris = namedtuple('iris', ['x', 'y', 'r'])(iris_coords[0], iris_coords[1], iris_coords[2])
        self.pupil = namedtuple('pupil', ['x', 'y', 'r'])(pupil_coords[0], pupil_coords[1], pupil_coords[2])
    
    def assert_iris_sclera_contrast_above_thresh(self, contrast):
        return contrast >= self.IRIS_SCLERA_THRESH
    
    def assert_iris_pupil_contrast_above_thresh(self, contrast):
        return contrast >= self.IRIS_PUPIL_THRESH
    
    def calculate_iris_sclera_contrast(self):
        iris_val = self.generate_iris_mask().value
        sclera_val = self.generate_sclera_mask().value
        pupil_val = self.generate_pupil_mask().value
        
        if (pupil_val >= iris_val) or (pupil_val >= sclera_val):
            return 0
        else:
            return 100 * abs(sclera_val - iris_val) / (sclera_val + iris_val - (2 * pupil_val))
        
    def calculate_iris_pupil_contrast(self):
        pupil_val =self.generate_pupil_mask().value
        iris_val = self.generate_iris_mask(pupil_calc=True).value
        
        weber_ratio = abs(iris_val - pupil_val)/(20 + pupil_val)
        return weber_ratio * 100 / (0.75 + weber_ratio)


    def generate_iris_mask(self, pupil_calc = False):
        if pupil_calc:
            return self.generate_area_mask(
                inner_radius = 1.1 * self.pupil.r,
                outer_radius = (self.pupil.r + self.iris.r) * 0.5,
                mask_name = "Iris area mask for pupil"
                )
        else:
            return self.generate_area_mask(
                inner_radius = (self.pupil.r + self.iris.r) * 0.5,
                outer_radius = 0.9 * self.iris.r,
                mask_name = "Iris area mask"
                )
    
    def generate_sclera_mask(self):
        return self.generate_area_mask(
            inner_radius = 1.1 * self.iris.r, 
            outer_radius = 1.2 * self.iris.r, 
            mask_name = "Sclera area mask"
            )
    
    def generate_pupil_mask(self, display_mask = False):
        mask = generate_empty_mask(self.img)
        cv2.circle(mask, (int(self.pupil.x), int(self.pupil.y)), int(0.8 * self.pupil.r), 255, -1)
        cv2.imshow("Pupil area mask", cv2.bitwise_and(self.img, self.img, mask=mask)) if display_mask else None
        
        value = self.__get_value(mask)
        
        return namedtuple('image_mask', ['mask', 'value'])(mask, value)

    def generate_area_mask(self, inner_radius, outer_radius, mask_name = "Area mask", display_mask = False):
        mask = generate_empty_mask(self.img)
        
        self.__draw_mask_bounds(mask, self.iris.x, self.iris.y, outer_radius)
        self.__draw_mask_bounds(mask, self.iris.x, self.iris.y, inner_radius, inner = True)      
        value = self.__get_value(mask)
        
        cv2.imshow(mask_name, cv2.bitwise_and(self.img, self.img, mask=mask)) if display_mask else None
        
        return namedtuple('image_mask', ['mask', 'value'])(mask, value)
    

    def __draw_mask_bounds(self, mask, x, y, r, inner = False):
        color = 0 if inner else 255
        angle = 35 if inner else 30
        
        cv2.ellipse(mask, (int(x), int(y)), (int(r), int(r)), 0, -angle, angle, color, -1)
        cv2.ellipse(mask, (int(x), int(y)), (int(r), int(r)), 180, -angle, angle, color, -1)
    
    def __get_value(self, mask):
        return np.median(self.img[mask == 255])
