# from sys import argv
# from sys import exit as sysexit

import cv2
import helper_functions as hf
import quality_metrics.required.grey_scale_utilization as gs
import quality_metrics.required.iris_pupil_concentricity as ipc
import quality_metrics.required.contrast as isc
import quality_metrics.required.iris_radius as ir
import quality_metrics.required.margin_adequacy as ma
import quality_metrics.required.pupil_boundary_circularity as pbc
import quality_metrics.required.pupil_dilation as pd
import quality_metrics.required.sharpness as sh
import quality_metrics.required.usable_iris_area as uia

if __name__ == "__main__":
    # if len(argv) == 1:
    #     print("Usage: python main.py <image path>")
    #     sysexit()
    
    print("System start")
    
    base_path = "resources/data_degraded"
        
    image_list = base_path + "/process_Grayscale.txt"
    image_path = base_path + "/Grayscale/"
    parameter_path = base_path + "/Output/CircleParameters/"
    mask_path = base_path + "/Output/Masks/"
    
    image_name = "256"
    image_extension = ".png"
    
    # img = cv2.imread(argv[1], cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(image_path + image_name + image_extension, cv2.IMREAD_GRAYSCALE)
    seg_mask = cv2.imread(mask_path + image_name + "_mask.bmp", cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (640, 480))
    
    fine_pupil_coords, fine_iris_coords = hf.read_OSIRIS_coords_from_file(parameter_path + image_name + "_para.txt")
    pupil_coords = hf.convert_OSIRIS_coords_to_xyr(fine_pupil_coords)
    iris_coords = hf.convert_OSIRIS_coords_to_xyr(fine_iris_coords)
    
    # hf.draw_iris_on_img(img, iris_coords)
    # hf.draw_pupil_on_img(img, pupil_coords)
    
    cv2.imshow("Original", img)
    
    uiaCalc = uia.UsableIrisAreaCalc(img, iris_coords, pupil_coords, seg_mask)
    iscCalc = isc.ContrastCalc(img, iris_coords, pupil_coords)
    gsCalc = gs.GreyScaleUtilizationCalc(img)
    irCalc = ir.IrisRadiusCalc(img, iris_coords)
    pdCalc = pd.PupilDilationCalc(img, pupil_coords, iris_coords)
    ipcCalc = ipc.IrisPupilCocentricityCalc(img, pupil_coords, iris_coords)
    maCalc = ma.MarginAdequacyCalc(img, iris_coords, pupil_coords)
    pbcCalc = pbc.PupilBoundaryCircularityCalc(img, pupil_coords, iris_coords)
    shCalc = sh.SharpnessCalc(img)
    
    print("=====================================")
    
    
    usable_area = uiaCalc.calculate_usable_area()
    print("Usable area: {}".format(round(usable_area, 3)))
    print("Usable area above threshold? {}".format(uiaCalc.assert_area_above_thresh(usable_area)))
    print("=====================================")
    
    iris_sclera_contrast = iscCalc.calculate_iris_sclera_contrast()
    print("Iris/Sclera contrast: {}".format(round(iris_sclera_contrast, 3)))
    print("Iris/Sclera contrast above threshold? {}".format(iscCalc.assert_iris_sclera_contrast_above_thresh(iris_sclera_contrast)))
    print("=====================================")
    
    iris_pupil_contrast = iscCalc.calculate_iris_pupil_contrast()
    print("Iris/Pupil contrast: {}".format(round(iris_pupil_contrast, 3)))
    print("Iris/Pupil contrast above threshold? {}".format(iscCalc.assert_iris_pupil_contrast_above_thresh(iris_pupil_contrast)))
    print("=====================================")
    
    entropy = gsCalc.calculate_entropy_in_bits()
    print("Entropy: {}".format(round(entropy, 3)))
    print("Entropy above threshold? {}".format(gsCalc.assert_entropy_above_thresh(entropy)))
    print("=====================================")
    
    iris_radius = irCalc.calculate_iris_radius()
    print("Iris radius: {}".format(round(iris_radius, 3)))
    print("Iris radius above threshold? {}".format(irCalc.assert_iris_radius_above_thresh(iris_radius)))
    print("=====================================")
    
    pupil_dilation = pdCalc.calculate_pupil_dilation()
    print("Pupil dilation: {}".format(round(pupil_dilation, 3)))
    print("Pupil dilation within thresholds? {}".format(pdCalc.assert_pupil_dilation_within_thresh(pupil_dilation)))
    print("=====================================")
    
    cocentricity = ipcCalc.calculate_cocentricity()
    print("Cocentricity: {}".format(round(cocentricity, 3)))
    print("Cocentricity above threshold? {}".format(ipcCalc.assert_cocentricity_above_thresh(cocentricity)))
    print("=====================================")
    
    margin_adequacy = maCalc.calculate_margin_adequacy()
    print("Margin adequacy: {}".format(round(margin_adequacy, 3)))
    print("Margin adequacy above threshold? {}".format(maCalc.assert_margin_adequacy_above_thresh(margin_adequacy)))
    print("=====================================")
    
    sharpness = shCalc.calculate_sharpness()
    print("Sharpness: {}".format(round(sharpness, 3)))
    #print("Sharpness above threshold? {}".format(shCalc.assert_sharpness_above_thresh(sharpness)))
    print("=====================================")

    if fine_pupil_coords is None:
        print("Fine pupil conteurs were not provided. Skipping pupil boundary circularity calculation.")


    cv2.waitKey(0)
    cv2.destroyAllWindows()
