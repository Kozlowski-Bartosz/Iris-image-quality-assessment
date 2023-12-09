# Iris Image Quality Asssessment

## Introduction
This repository contains the code for the thesis "Ocular image quality assessment for the purposes of iris recognition" by Bartosz Kozłowski, supervised by dr inż. Wojciech Wodo. The thesis was written at the Faculty of Information Technology and Telecomunication (W4N), Wroclaw University of Science and Technology.

The program is able to assess the quality of an iris image following the ten recommended metrics from the ISO/IEC 29794-6:2015 standard, which are:
- Usable iris area
- Iris-sclera contrast
- Iris-pupil contrast
- Pupil boundary circularity
- Grey-scale utilization
- Iris radius
- Pupil dilation
- Iris pupil concentricity
- Margin adequacy
- Sharpness

## Instructions

The parameter files need to follow the naming convention of: `<image_name_without_extension><params_suffix>` (For example: `001_1_1_para.txt`, where `001_1_1` is the image name, and `_para.txt` was configured as the parameter suffix in the config.ini file). Similarly, the mask files need to follow the naming convention of: `<image_name_without_extension><mask_suffix>` (For example: `001_1_1_mask.bmp`). The mask files need to be binary images, where 1 represents the iris and 0 represents the background. The mask files need to be the same size (width and height) as the corresponding image.