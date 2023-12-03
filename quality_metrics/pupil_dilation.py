from collections import namedtuple

class PupilDilationCalc:
    
    def __init__(self, img, pupil_coords, iris_coords):
        self.img = img
        self.pupil = namedtuple('pupil', ['x', 'y', 'r'])(pupil_coords[0], pupil_coords[1], pupil_coords[2])
        self.iris = namedtuple('iris', ['x', 'y', 'r'])(iris_coords[0], iris_coords[1], iris_coords[2])
        
    def calculate_pupil_dilation(self):
        return 100 * self.pupil.r / self.iris.r