import cv2
import numpy as np
import math

class SharpnessCalc:
    
    def __init__(self, img):
        self.img = img
        
    def calculate_sharpness(self):
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
        
        k_width, k_height = kernel.shape[:2]
        step = 4
        
        adj_width = int((width - k_width) / step + 1)
        adj_height = int((height - k_height) / step + 1)
        
        filtered_img = np.zeros((adj_width, adj_height))
        
        rows = np.arange(4, width-4, 4)
        cols = np.arange(4, height-4, 4)
        
        for x in rows:
            for y in cols:
                # Extract a 9x9 region of interest
                roi = self.img[x-4:x+4+1, y-4:y+4+1]
                # Apply the kernel to the region of interest
                sum_of_pixels = np.sum(np.multiply(roi, kernel))
                # print(int(x/4), int(y/4))
                filtered_img[int(x/4 - 1), int(y/4 - 1)] = sum_of_pixels
                
        # For each pixel in filtered image, square it. Sum all the pixels.
        squares_sum = np.sum(np.square(filtered_img))
        
        power = squares_sum / (width * height)
        power_sq = pow(power, 2)
        
        C_SQUARED = 3240000000000
        
        sharpness = (power_sq / (power_sq + C_SQUARED)) * 100
        cv2.imshow("Sharpness", filtered_img)

        return sharpness
        