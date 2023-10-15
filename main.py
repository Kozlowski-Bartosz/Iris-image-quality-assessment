from sys import argv
from sys import exit as sysexit

import cv2
# import quality_metrics.required.grey_scale_utilization
# import quality_metrics.required.iris_pupil_cocentricity
# import quality_metrics.required.iris_sclera_contrast
# import quality_metrics.required.iris_pupil_contrast
# import quality_metrics.required.iris_radius
# import quality_metrics.required.margin_adequacy
# import quality_metrics.required.pupil_boundary_circularity
# import quality_metrics.required.pupil_dilation
# import quality_metrics.required.sharpness
# import quality_metrics.required.usable_iris_area


if len(argv) == 1:
    print("Usage: python main.py <image path>")
    sysexit()

img = cv2.imread(argv[1])


cv2.waitKey(0)
cv2.destroyAllWindows()