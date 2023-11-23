import cv2
from collections import namedtuple

class sharpnessCalc:
    
    def __init__(self, img):
        self.img = img
        
    def calculate_sharpness(self):
        F = [
            [0, 1, 1, 2, 2, 2, 1, 1, 0],
            [1, 2, 4, 5, 5, 5, 4, 2, 1],
            [1, 4, 5, 3, 0 ,3 ,5, 4, 1],
            [2, 5, 3, -12, -24, -12, 3, 5, 2],
            [2, 5, 0, -24, -40, -24, 0, 5, 2],
            [2, 5, 3, -12, -24, -12, 3, 5, 2],
            [1, 4, 5, 3, 0 ,3 ,5, 4, 1],
            [1, 2, 4, 5, 5, 5, 4, 2, 1],
            [0, 1, 1, 2, 2, 2, 1, 1, 0]
            ]
