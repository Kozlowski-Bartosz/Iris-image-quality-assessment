import numpy as np
import cv2
from scipy.optimize import least_squares

def read_OSIRIS_coords_from_file(filepath):
    try:
        with open(filepath, 'r') as file:
            # Read all lines into a list
            lines = file.readlines()

            # Check if there are at least four lines in the file
            if len(lines) < 4:
                return None

            # Split the third line into space-separated values
            pupil_values = lines[2].strip().split()
            iris_values = lines[3].strip().split()


            # Group the values into sets of two
            grouped_pupil_values = [pupil_values[i:i+2] for i in range(0, len(pupil_values), 3)]
            grouped_iris_values = [iris_values[i:i+2] for i in range(0, len(iris_values), 3)]
            
            return grouped_pupil_values, grouped_iris_values
        
    except FileNotFoundError:
        print(f"File '{filepath}' not found.")
        return None
    
def read_classical_coords_from_file(filepath):
    try:
        with open(filepath, 'r') as file:
            # Read all lines into a list
            lines = file.readlines()
            line_count = len(lines)

            pupil_values = np.array(lines[0].strip().split(), dtype=np.float64) if line_count >= 1 else None
            iris_values = np.array(lines[1].strip().split(), dtype=np.float64) if line_count >= 2 else None
            pupil_fine_coords = np.array(lines[2].strip().split(), dtype=np.float64) if line_count >= 3 else None
            iris_fine_coords = np.array(lines[0].strip().split(), dtype=np.float64) if line_count >= 4 else None
            
            grouped_pupil_fine_coords = [pupil_fine_coords[i:i+2] for i in range(0, len(pupil_fine_coords), 2)] if line_count >= 3 else None
            grouped_iris_fine_coords = [iris_fine_coords[i:i+2] for i in range(0, len(iris_fine_coords), 2)] if line_count >= 4 else None
            
            return pupil_values, iris_values, grouped_pupil_fine_coords, grouped_iris_fine_coords
        
    except FileNotFoundError:
        print(f"File '{filepath}' not found.")
        return None
    
def circle_equation(params, x, y):
    a, b, r = params
    return (x - a) ** 2 + (y - b) ** 2 - r ** 2

def convert_OSIRIS_coords_to_xyr(coords):
    coords = np.array(coords, dtype=np.float64)
    initial_guess = (0, 0, 1)

    # Fit the circle parameters using the least squares method
    result = least_squares(circle_equation, initial_guess, args=(coords[:, 0], coords[:, 1]))

    # Extract the center coordinates (a, b) and radius (r)
    center_x, center_y, radius = result.x
    radius = abs(radius)
    return center_x, center_y, radius

def draw_pupil_on_img(img, pupil_coords):
    pupil_x, pupil_y, pupil_radius = pupil_coords
    cv2.circle(img, (int(pupil_x), int(pupil_y)), int(pupil_radius), (255, 0, 0), 1)
    return img

def draw_iris_on_img(img, iris_coords):
    iris_x, iris_y, iris_radius = iris_coords
    cv2.circle(img, (int(iris_x), int(iris_y)), int(iris_radius), (255, 0, 0), 1)
    return img

def generate_empty_mask(img):
    return np.zeros(img.shape[:2], dtype=np.uint8)

def assert_within_threshold(value, threshold):
    within_lower = (value >= threshold[0]) if threshold[0] is not None else True
    within_upper = (value <= threshold[1]) if threshold[1] is not None else True
    return within_lower and within_upper