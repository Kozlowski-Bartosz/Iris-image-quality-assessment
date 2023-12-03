import cv2
import numpy as np
from collections import namedtuple

class PupilBoundaryCircularityCalc:
    
    def __init__(self, img, pupil_coords, fine_pupil_coords):
        self.img = img
        self.pupil = namedtuple('pupil', ['x', 'y', 'r'])(pupil_coords[0], pupil_coords[1], pupil_coords[2])
        self.fine_pupil = np.array(fine_pupil_coords)
        self.fine_x = self.fine_pupil[:,0]
        self.fine_y = self.fine_pupil[:,1]
        
    def calculate_circularity(self):
        n = len(self.fine_pupil[:])
        m = 17
        r_array = np.array(self.define_radii()).astype(complex)
        # print(r_array)
        C = np.zeros(m).astype(complex)
        
        modulus_sum = 0
        for k in range(1, m):
            for theta in range(n):
                C[k] += r_array[theta] * np.exp(-2 * np.pi * 1j * k * theta / n)
                
            modulus_sum += abs(C[k]) ** 2
                
        calculated_circularity = 100.0 - (modulus_sum/n)
        return(max(0.0, calculated_circularity))        
        
    def is_point_occluded(self, x, y, mask):
        return mask[x][y] == 0
          
    def define_radii(self):
        
        pupil_radii = []
        for i in range(len(self.fine_pupil)):
            radius = self.calculate_distance(self.pupil.x, self.pupil.y, float(self.fine_x[i]), float(self.fine_y[i]))
            pupil_radii.append(radius)
        return pupil_radii
            
    def calculate_distance(self, x1, y1, x2, y2):
        return np.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    