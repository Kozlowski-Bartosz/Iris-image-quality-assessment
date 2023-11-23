import cv2
import numpy as np
from collections import namedtuple

class PupilBoundaryCircularityCalc:
    
    def __init__(self, img, pupil_coords, fine_pupil_coords):
        self.img = img
        self.pupil = namedtuple('pupil', ['x', 'y', 'r'])(pupil_coords[0], pupil_coords[1], pupil_coords[2])
        self.fine_pupil = namedtuple('fine_pupil', ['x', 'y', 'phi'])(fine_pupil_coords[0], fine_pupil_coords[1], fine_pupil_coords[2])
        
      
      
    def calculate_circularity(self):
        N = len(self.fine_pupil)
        
        
          
    # def define_radii(self):
        
    #     pupil_radii = []
    #     for i in range(len(self.fine_pupil)):
    #         radius = self.calculate_distance(self.pupil.x, self.pupil.y, self.fine_pupil[i].x, self.fine_pupil[i].y)
    #         pupil_radii.append(radius)
            
    # def calculate_distance(self, x1, y1, x2, y2):
    #     return np.sqrt((x2-x1)**2 + (y2-y1)**2)