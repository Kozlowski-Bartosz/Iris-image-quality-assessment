import math
import cv2

class GreyScaleUtilizationCalc:
    
    THRESH = 6.0
    
    def __init__(self, img):
        self.img = img
        
    def calculate_entropy_in_bits(self):
        entropy = 0
        grey_level_count = self.count_grey_levels()
        
        for level in range(grey_level_count):
            p = self.calculate_probability(level)
            entropy += -p * math.log(p, 2) if p > 0 else 0
        
        return entropy
        
    def count_grey_levels(self):
        return len(set(self.img.flatten()))
    
    def calculate_probability(self, level):
        px_in_level = (self.img[self.img == level])
        return len(px_in_level) / self.img.size
    
    def assert_entropy_above_thresh(self, entropy):
        return entropy >= self.THRESH
    
    def normalize_entropy_score(self, entropy):
        return entropy / self.THRESH * 100