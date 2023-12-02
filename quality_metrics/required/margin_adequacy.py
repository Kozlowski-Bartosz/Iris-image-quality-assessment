from collections import namedtuple
import cv2

class MarginAdequacyCalc:
    
    THRESH = 80.0
    
    def __init__(self, img, iris_coords, pupil_coords):
        self.img = img
        self.iris = namedtuple('iris', ['x', 'y', 'r'])(iris_coords[0], iris_coords[1], iris_coords[2])
        self.pupil = namedtuple('pupil', ['x', 'y', 'r'])(pupil_coords[0], pupil_coords[1], pupil_coords[2])
        
    def calculate_margin_adequacy(self):
        image_width = self.img.shape[1]
        image_height = self.img.shape[0]
        LM = (self.iris.x - self.iris.r)/self.iris.r
        RM = (image_width - (self.iris.x + self.iris.r))/self.iris.r
        DM = (image_height - (self.iris.y + self.iris.r))/self.iris.r
        UM = (self.iris.y - self.iris.r)/self.iris.r
        
        LEFT_MARGIN = max(0, min(1, LM/0.6))
        RIGHT_MARGIN = max(0, min(1, RM/0.6))
        UP_MARGIN = max(0, min(1, UM/0.2))
        DOWN_MARGIN = max(0, min(1, DM/0.2))
        
        # cv2.imshow("Margin adequacy",self.draw_margins())
        
        return 100 * min(LEFT_MARGIN, RIGHT_MARGIN, UP_MARGIN, DOWN_MARGIN)
    
    def draw_margins(self):
        image_width = self.img.shape[1]
        image_height = self.img.shape[0]
        
        left_margin = int(self.iris.x - self.iris.r)
        right_margin = int(self.iris.x + self.iris.r)
        up_margin = int(self.iris.y - self.iris.r)
        down_margin = int(self.iris.y + self.iris.r)
        
        cv2.line(self.img, (left_margin, 0), (left_margin, image_height), (0, 0, 255), 1)
        cv2.line(self.img, (right_margin, 0), (right_margin, image_height), (0, 0, 255), 1)
        cv2.line(self.img, (0, up_margin), (image_width, up_margin), (0, 0, 255), 1)
        cv2.line(self.img, (0, down_margin), (image_width, down_margin), (0, 0, 255), 1)
        
        return self.img
        
        
    def assert_margin_adequacy_above_thresh(self, margin_adequacy):
        return margin_adequacy >= self.THRESH