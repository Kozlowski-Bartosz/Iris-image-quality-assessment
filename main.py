# from sys import argv
# from sys import exit as sysexit

import time
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


if __name__ == "__main__":
       
    try:
        cf = Config("config.ini")
    except KeyError:
        print("Config file not found or incorrectly formated.")
        exit()    
        
    start_time = time.time()
    image_count = len(cf.images)
    
    print("Starting quality metrics calculations.")
        
    for i in range(image_count):
        print("Processing image {} of {}.".format(i+1, image_count))
        try:
            img = cv2.imread(cf.image_path + cf.image_names[i] + cf.image_extensions[i], cv2.IMREAD_GRAYSCALE)
            seg_mask = cv2.imread(cf.mask_path + cf.image_names[i] + cf.mask_suffix, cv2.IMREAD_GRAYSCALE)
        except:
            print("Image file read fail for image {}.".format(cf.image_names[i]))
            continue
        
        if cf.convert_from_OSIRIS:
            try:
                fine_pupil_coords, fine_iris_coords = hf.read_OSIRIS_coords_from_file(cf.parameter_path + cf.image_names[i] + cf.parameters_suffix)
                pupil_coords = hf.convert_OSIRIS_coords_to_xyr(fine_pupil_coords)
                iris_coords = hf.convert_OSIRIS_coords_to_xyr(fine_iris_coords)
            except TypeError:
                print("Parameter file not found for image {}.".format(cf.image_names[i]))
                continue
        else:
            pupil_coords, iris_coords, fine_pupil_coords, fine_iris_coords = hf.read_classical_coords_from_file(cf.parameter_path + cf.image_names[i] + "_para.txt")
        params = []
        
        if cf.UsableIrisArea:
            try:
                uiaCalc = uia.UsableIrisAreaCalc(img, iris_coords, pupil_coords, seg_mask)
                usable_area = uiaCalc.calculate_usable_area()
                usable_area_thresh = hf.assert_within_threshold(usable_area, cf.thresh_uia) if cf.check_thresholds else True
                params.append(["Usable_Iris_Area", usable_area, usable_area_thresh])
            except:
                print("Usable Iris Area calculation failed. Skipping calculation.")
        
        if cf.IrisScleraContrast:
            try:
                iscCalc = isc.ContrastCalc(img, iris_coords, pupil_coords)
                iris_sclera_contrast = iscCalc.calculate_iris_sclera_contrast()
                iris_sclera_contrast_thresh = hf.assert_within_threshold(iris_sclera_contrast, cf.thresh_isc) if cf.check_thresholds else True
                params.append(["Iris_Sclera_Contrast", iris_sclera_contrast, iris_sclera_contrast_thresh])
            except:
                print("Iris Sclera Contrast calculation failed. Skipping calculation.")
        
        if cf.IrisPupilContrast:
            try:
                iscCalc = isc.ContrastCalc(img, iris_coords, pupil_coords) if not cf.IrisScleraContrast else iscCalc
                iris_pupil_contrast = iscCalc.calculate_iris_pupil_contrast()
                iris_pupil_contrast_thresh = hf.assert_within_threshold(iris_pupil_contrast, cf.thresh_ipc) if cf.check_thresholds else True
                params.append(["Iris_Pupil_Contrast", iris_pupil_contrast, iris_pupil_contrast_thresh])
            except:
                print("Iris Pupil Contrast calculation failed. Skipping calculation.")
        
        if cf.PupilBoundaryCircularity:
            try:
                pbcCalc = pbc.PupilBoundaryCircularityCalc(img, pupil_coords, fine_pupil_coords)
                pupil_circularity = pbcCalc.calculate_circularity()
                pupil_circularity_thresh = hf.assert_within_threshold(pupil_circularity, cf.thresh_pbc) if cf.check_thresholds else True
                params.append(["Pupil_Boundary_Circularity", pupil_circularity, pupil_circularity_thresh])
            except:
                print("Pupil Boundary Circularity calculation failed. Skipping calculation.")
        
        if cf.GreyScaleUtilization:
            try:
                gsCalc = gs.GreyScaleUtilizationCalc(img)
                entropy = gsCalc.calculate_entropy_in_bits()
                entropy_thresh = hf.assert_within_threshold(entropy, cf.thresh_gsu) if cf.check_thresholds else True
                params.append(["Grey_Scale_Utilization", entropy, entropy_thresh])
            except:
                print("Grey Scale Utilization calculation failed. Skipping calculation.")
        
        if cf.IrisRadius:
            try:
                irCalc = ir.IrisRadiusCalc(img, iris_coords)
                iris_radius = irCalc.calculate_iris_radius()
                iris_radius_thresh = hf.assert_within_threshold(iris_radius, cf.thresh_ir) if cf.check_thresholds else True
                params.append(["Iris_Radius", iris_radius, iris_radius_thresh])
            except:
                print("Iris Radius calculation failed. Skipping calculation.")
        
        if cf.PupilDilation:
            try:
                pdCalc = pd.PupilDilationCalc(img, pupil_coords, iris_coords)
                pupil_dilation = pdCalc.calculate_pupil_dilation()
                pupil_dilation_thresh = hf.assert_within_threshold(pupil_dilation, cf.thresh_pd) if cf.check_thresholds else True
                params.append(["Pupil_Dilation", pupil_dilation, pupil_dilation_thresh])
            except:
                print("Pupil Dilation calculation failed. Skipping calculation.")
        
        if cf.IrisPupilConcentricity:
            try:
                ipcCalc = ipc.IrisPupilCocentricityCalc(img, pupil_coords, iris_coords)
                cocentricity = ipcCalc.calculate_cocentricity()
                cocentricity_thresh = hf.assert_within_threshold(cocentricity, cf.thresh_ipcon) if cf.check_thresholds else True
                params.append(["Iris_Pupil_Concentricity", cocentricity, cocentricity_thresh])
            except:
                print("Iris Pupil Concentricity calculation failed. Skipping calculation.")
        
        if cf.MarginAdequacy:
            try:
                maCalc = ma.MarginAdequacyCalc(img, iris_coords, pupil_coords)
                margin_adequacy = maCalc.calculate_margin_adequacy()
                margin_adequacy_thresh = hf.assert_within_threshold(margin_adequacy, cf.thresh_ma) if cf.check_thresholds else True
                params.append(["Margin_Adequacy", margin_adequacy, margin_adequacy_thresh])
            except:
                print("Margin Adequacy calculation failed. Skipping calculation.")

        if cf.Sharpness:
            try:
                shCalc = sh.SharpnessCalc(img)
                sharpness = shCalc.calculate_sharpness()
                sharpness_thresh = hf.assert_within_threshold(sharpness, cf.thresh_sh) if cf.check_thresholds else True
                params.append(["Sharpness", sharpness, sharpness_thresh])
            except:
                print("Sharpness calculation failed. Skipping calculation.")

        if cf.output_to_console:
            print("Image: {}".format(cf.image_names[i]))
            print(65*"=") if cf.check_thresholds else print(40*"=")
            for param in params:
                if cf.check_thresholds:
                    print("{:<30}{:<15}Within thresh: {}".format(param[0], round(param[1], cf.round_to), param[2]))
                else:
                    print("{:<30}{}".format(param[0], round(param[1], cf.round_to)))
            print(65*"=") if cf.check_thresholds else print(40*"=")

            
        if cf.output_to_file and cf.output_file_path is not None:
            if i == 0:
                with open(cf.output_file_path, 'w') as f:
                    f.write("Image,")
                    for param in params:
                        f.write("{},".format(param[0]))
                        
                    if cf.check_thresholds:
                        for param in params:
                            f.write("{},".format(param[0]+"_thresh"))
            
            with open(cf.output_file_path, 'a') as f:    
                f.write("\n")
                f.write("{},".format(cf.image_names[i]))
                for param in params:
                    f.write("{},".format(round(param[1], cf.round_to)))
                    
                if cf.check_thresholds:
                    for param in params:
                        f.write("{},".format(param[2]))
                
                
                
                
                
    
    print("Execution ended after: {} seconds".format(time.time() - start_time))
    print("Average time per image: {} seconds".format((time.time() - start_time) / len(cf.images)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    input("Press the Enter key to continue: ") 