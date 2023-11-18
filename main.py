from sys import argv
from sys import exit as sysexit

import cv2
import numpy as np
import helper_functions as hf
import quality_metrics.required.grey_scale_utilization as gs
# import quality_metrics.required.iris_pupil_cocentricity
import quality_metrics.required.contrast as isc
# import quality_metrics.required.iris_pupil_contrast
# import quality_metrics.required.iris_radius
# import quality_metrics.required.margin_adequacy
# import quality_metrics.required.pupil_boundary_circularity
# import quality_metrics.required.pupil_dilation
# import quality_metrics.required.sharpness
import quality_metrics.required.usable_iris_area as uia

if __name__ == "__main__":
    # if len(argv) == 1:
    #     print("Usage: python main.py <image path>")
    #     sysexit()
        
    image_list = "resources/data_ubiris/process_UBIRIS.txt"
    image_path = "resources/data_ubiris/UBIRIS/"
    parameter_path = "resources/data_ubiris/Output/CircleParameters/"
    mask_path = "resources/data_ubiris/Output/Masks/"
    
    # img = cv2.imread(argv[1], cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(image_path + "2.jpg", cv2.IMREAD_GRAYSCALE)
    seg_mask = cv2.imread(mask_path + "2_mask.bmp", cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (640, 480))
    
    pupil_coords, iris_coords = hf.read_OSIRIS_coords_from_file(parameter_path + "2_para.txt")
    hf.draw_iris_on_img(img, iris_coords)
    hf.draw_pupil_on_img(img, pupil_coords)
    
    cv2.imshow("Original", img)
    
    uiaCalc = uia.UsableIrisAreaCalc(img, iris_coords, pupil_coords, seg_mask)
    usable_area = uiaCalc.calculate_usable_area()
    print("Usable area: {}".format(usable_area))
    print("Usable area above threshold? {}".format(uiaCalc.assert_area_above_thresh(usable_area)))
    
    iscCalc = isc.ContrastCalc(img, iris_coords, pupil_coords)
    print("Iris/Sclera contrast: {}".format(iscCalc.calculate_iris_sclera_contrast()))
    print("Iris/Pupil contrast: {}".format(iscCalc.calculate_iris_pupil_contrast()))
    
    gsCalc = gs.GreyScaleUtilizationCalc(img)
    entropy = gsCalc.calculate_entropy_in_bits()
    print("Entropy: {}".format(entropy))
    print("Entropy above threshold? {}".format(gsCalc.assert_entropy_above_thresh(entropy)))


    cv2.waitKey(0)
    cv2.destroyAllWindows()
