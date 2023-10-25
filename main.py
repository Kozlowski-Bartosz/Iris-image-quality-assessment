from sys import argv
from sys import exit as sysexit

import cv2
import numpy as np
import quality_metrics.required.grey_scale_utilization as gs
# import quality_metrics.required.iris_pupil_cocentricity
# import quality_metrics.required.iris_sclera_contrast
# import quality_metrics.required.iris_pupil_contrast
# import quality_metrics.required.iris_radius
# import quality_metrics.required.margin_adequacy
# import quality_metrics.required.pupil_boundary_circularity
# import quality_metrics.required.pupil_dilation
# import quality_metrics.required.sharpness
# import quality_metrics.required.usable_iris_area

if __name__ == "__main__":
    if len(argv) == 1:
        print("Usage: python main.py <image path>")
        sysexit()

    img = cv2.imread(argv[1], cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (320, 240))
    cv2.imshow("Image", img)
    
    grey = gs.GreyScale(img)
    entropy = grey.calculate_entropy_in_bits()
    print("Entropy: {}".format(entropy))
    print(grey.is_entropy_above_thresh(entropy))


    cv2.waitKey(0)
    cv2.destroyAllWindows()