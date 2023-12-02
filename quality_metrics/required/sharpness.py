import cv2
from collections import namedtuple
import numpy as np

class SharpnessCalc:
    
    def __init__(self, img):
        self.img = img
        
    def calculate_sharpness(self):
        print("Calculating sharpness...")
        width, height = self.img.shape[:2]
        
        kernel = np.array([
            [0, 1, 1, 2, 2, 2, 1, 1, 0],
            [1, 2, 4, 5, 5, 5, 4, 2, 1],
            [1, 4, 5, 3, 0 ,3 ,5, 4, 1],
            [2, 5, 3, -12, -24, -12, 3, 5, 2],
            [2, 5, 0, -24, -40, -24, 0, 5, 2],
            [2, 5, 3, -12, -24, -12, 3, 5, 2],
            [1, 4, 5, 3, 0 ,3 ,5, 4, 1],
            [1, 2, 4, 5, 5, 5, 4, 2, 1],
            [0, 1, 1, 2, 2, 2, 1, 1, 0]
        ])
        
        filtered_img = np.zeros_like(self.img)
        
        rows = np.arange(4, width-4, 4)
        cols = np.arange(4, height-4, 4)
        
        # Perform convolution on the image every 4 rows and columns
        for x in rows:
            for y in cols:
                # Extract a 9x9 region of interest
                roi = self.img[x-4:x+5, y-4:y+5]
                # Apply the kernel to the region of interest
                filtered_img[x-4:x+5, y-4:y+5] = np.multiply(roi, kernel)
        
        cv2.imshow("Sharpness", filtered_img)
