from collections import namedtuple

class IrisRadiusCalc:
    
    def __init__(self, img, iris_coords):
        self.img = img
        self.iris = namedtuple('iris', ['x', 'y', 'r'])(iris_coords[0], iris_coords[1], iris_coords[2])
        
    def calculate_iris_radius(self):
        return self.iris.r