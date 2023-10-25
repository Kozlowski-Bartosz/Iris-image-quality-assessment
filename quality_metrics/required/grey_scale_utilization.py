import math
import cv2

class GreyScale:
    
    THRESH = 6.0
    
    def __init__(self, image):
        self.image = image
        
    def get_grey_level_count_in_image(self):
        return len(set(self.image.flatten()))
        
    def calculate_entropy_in_bits(self):
        entropy = 0
        grey_level_count = self.get_grey_level_count_in_image()
        px_in_img = self.image.size
        
        for level in range(grey_level_count):
            px_in_level = (self.image[self.image == level])
            p = len(px_in_level) / px_in_img
            entropy += -p * math.log(p, 2) if p > 0 else 0
            
        return entropy
    
    def is_entropy_above_thresh(self, entropy):
        assert entropy >= self.THRESH, "Entropy is below threshold."