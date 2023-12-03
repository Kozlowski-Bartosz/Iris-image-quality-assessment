# from sys import argv
# from sys import exit as sysexit

import cv2
import helper_functions as hf
from config_parser import Config
import quality_metrics.grey_scale_utilization as gs
import quality_metrics.iris_pupil_concentricity as ipc
import quality_metrics.contrast as isc
import quality_metrics.iris_radius as ir
import quality_metrics.margin_adequacy as ma
import quality_metrics.pupil_boundary_circularity as pbc
import quality_metrics.pupil_dilation as pd
import quality_metrics.sharpness as sh
import quality_metrics.usable_iris_area as uia

import time

if __name__ == "__main__":
    try:
        cf = Config("config.ini")
    except KeyError:
        print("Config file not found or incorrectly formated.")
        exit()
        
    start_time = time.time()
    print("Starting quality metrics calculations.")
        
    for i in range(len(cf.images)):
        
        try:
            img = cv2.imread(cf.image_path + cf.image_names[i] + cf.image_extensions[i], cv2.IMREAD_GRAYSCALE)
            seg_mask = cv2.imread(cf.mask_path + cf.image_names[i] + "_mask.bmp", cv2.IMREAD_GRAYSCALE)
        except:
            continue
        
        if cf.convert_from_OSIRIS:
            try:
                fine_pupil_coords, fine_iris_coords = hf.read_OSIRIS_coords_from_file(cf.parameter_path + cf.image_names[i] + "_para.txt")
                pupil_coords = hf.convert_OSIRIS_coords_to_xyr(fine_pupil_coords)
                iris_coords = hf.convert_OSIRIS_coords_to_xyr(fine_iris_coords)
            except TypeError:
                print("Parameter file not found for image {}.".format(cf.image_names[i]))
                continue   
        
        #print("Processing image: {}".format(cf.image_names[i]))
        
        if cf.UsableIrisArea:
            try:
                uiaCalc = uia.UsableIrisAreaCalc(img, iris_coords, pupil_coords, seg_mask)
                usable_area = uiaCalc.calculate_usable_area()
                usable_area_thresh = hf.assert_within_threshold(usable_area, cf.thresh_uia) if cf.check_thresholds else True
            except:
                print("Usable Iris Area calculation failed. Skipping calculation.")
        
        if cf.IrisScleraContrast:
            try:
                iscCalc = isc.ContrastCalc(img, iris_coords, pupil_coords)
                iris_sclera_contrast = iscCalc.calculate_iris_sclera_contrast()
                iris_sclera_contrast_thresh = hf.assert_within_threshold(iris_sclera_contrast, cf.thresh_isc) if cf.check_thresholds else True
            except:
                print("Iris Sclera Contrast calculation failed. Skipping calculation.")
        
        if cf.IrisPupilContrast:
            try:
                iscCalc = isc.ContrastCalc(img, iris_coords, pupil_coords) if not cf.IrisScleraContrast else iscCalc
                iris_pupil_contrast = iscCalc.calculate_iris_pupil_contrast()
                iris_pupil_contrast_thresh = hf.assert_within_threshold(iris_pupil_contrast, cf.thresh_ipc) if cf.check_thresholds else True
            except:
                print("Iris Pupil Contrast calculation failed. Skipping calculation.")
        
        if cf.PupilBoundaryCircularity:
            try:
                pbcCalc = pbc.PupilBoundaryCircularityCalc(img, pupil_coords, fine_pupil_coords)
                pupil_circularity = pbcCalc.calculate_circularity()
                pupil_circularity_thresh = hf.assert_within_threshold(pupil_circularity, cf.thresh_pbc) if cf.check_thresholds else True
            except:
                print("Pupil Boundary Circularity calculation failed. Skipping calculation.")
        
        if cf.GreyScaleUtilization:
            try:
                gsCalc = gs.GreyScaleUtilizationCalc(img)
                entropy = gsCalc.calculate_entropy_in_bits()
                entropy_thresh = hf.assert_within_threshold(entropy, cf.thresh_gsu) if cf.check_thresholds else True
            except:
                print("Grey Scale Utilization calculation failed. Skipping calculation.")

        
        if cf.IrisRadius:
            try:
                irCalc = ir.IrisRadiusCalc(img, iris_coords)
                iris_radius = irCalc.calculate_iris_radius()
                iris_radius_thresh = hf.assert_within_threshold(iris_radius, cf.thresh_ir) if cf.check_thresholds else True
            except:
                print("Iris Radius calculation failed. Skipping calculation.")
        
        if cf.PupilDilation:
            try:
                pdCalc = pd.PupilDilationCalc(img, pupil_coords, iris_coords)
                pupil_dilation = pdCalc.calculate_pupil_dilation()
                pupil_dilation_thresh = hf.assert_within_threshold(pupil_dilation, cf.thresh_pd) if cf.check_thresholds else True
            except:
                print("Pupil Dilation calculation failed. Skipping calculation.")

        
        if cf.IrisPupilConcentricity:
            try:
                ipcCalc = ipc.IrisPupilCocentricityCalc(img, pupil_coords, iris_coords)
                cocentricity = ipcCalc.calculate_cocentricity()
                cocentricity_thresh = hf.assert_within_threshold(cocentricity, cf.thresh_ipcon) if cf.check_thresholds else True
            except:
                print("Iris Pupil Concentricity calculation failed. Skipping calculation.")

        
        if cf.MarginAdequacy:
            try:
                maCalc = ma.MarginAdequacyCalc(img, iris_coords, pupil_coords)
                margin_adequacy = maCalc.calculate_margin_adequacy()
                margin_adequacy_thresh = hf.assert_within_threshold(margin_adequacy, cf.thresh_ma) if cf.check_thresholds else True
            except:
                print("Margin Adequacy calculation failed. Skipping calculation.")

        
        if cf.Sharpness:
            try:
                shCalc = sh.SharpnessCalc(img)
                sharpness = shCalc.calculate_sharpness()
                sharpness_thresh = hf.assert_within_threshold(sharpness, cf.thresh_sh) if cf.check_thresholds else True
            except:
                print("Sharpness calculation failed. Skipping calculation.")

        if cf.output_to_console:
            print("Image: {}".format(cf.image_names[i]))
            print("=====================================================")
            if cf.UsableIrisArea:
                print("Usable Iris Area:\t\t{}\t\tWithin thresh: {}".format(round(usable_area, 3), usable_area_thresh))
            if cf.IrisScleraContrast:
                print("Iris Sclera Contrast:\t\t{}\t\tWithin thresh: {}".format(round(iris_sclera_contrast, 3), iris_sclera_contrast_thresh))
            if cf.IrisPupilContrast:
                print("Iris Pupil Contrast:\t\t{}\t\tWithin thresh: {}".format(round(iris_pupil_contrast, 3), iris_pupil_contrast_thresh))
            if cf.PupilBoundaryCircularity:
                print("Pupil Boundary Circularity:\t{}\t\tWithin thresh: {}".format(round(pupil_circularity, 3), pupil_circularity_thresh))
            if cf.GreyScaleUtilization:
                print("Grey Scale Utilization:\t\t{}\t\tWithin thresh: {}".format(round(entropy, 3), entropy_thresh))
            if cf.IrisRadius:
                print("Iris Radius:\t\t\t{}\t\tWithin thresh: {}".format(round(iris_radius, 3), iris_radius_thresh))
            if cf.PupilDilation:
                print("Pupil Dilation:\t\t\t{}\t\tWithin thresh: {}".format(round(pupil_dilation, 3), pupil_dilation_thresh))
            if cf.IrisPupilConcentricity:
                print("Iris Pupil Concentricity:\t{}\t\tWithin thresh: {}".format(round(cocentricity, 3), cocentricity_thresh))
            if cf.MarginAdequacy:
                print("Margin Adequacy:\t\t{}\t\tWithin thresh: {}".format(round(margin_adequacy), margin_adequacy_thresh))
            if cf.Sharpness:
                print("Sharpness:\t\t\t{}\t\tWithin thresh: {}".format(round(sharpness, 3), sharpness_thresh))
            print("=====================================================")
    
    print("Quality metrics calculations complete. End time: {} seconds".format(time.time() - start_time))
    print("Average time per image: {} seconds".format((time.time() - start_time) / len(cf.images)))