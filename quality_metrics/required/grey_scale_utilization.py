import math
import cv2

THRESH = 6.0

class GreyScale:
    def __init__(self, image):
        self.image = image
        
    def calculate_entropy(self):
        entropy = 0
        # function here
        return entropy
    
    def is_entropy_above_thresh(self, THRESH):
        entropy = self.calculate_entropy()
        if entropy >= THRESH:
            return True
        else:
            return False