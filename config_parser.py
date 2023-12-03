import configparser
import pathlib

#config = configparser.ConfigParser()

class Config(configparser.ConfigParser):
    
    def __init__(self, config_file):
        super().__init__()
        self.read(config_file)

        self.image_list = self['Paths']['image_list_path']
        self.image_path = self['Paths']['image_directory_path']
        self.parameter_path = self['Paths']['parameters_path']
        self.mask_path = self['Paths']['mask_path']
        
        self.UsableIrisArea = self['QualityMetrics'].getboolean('UsableIrisArea')
        self.IrisScleraContrast = self['QualityMetrics'].getboolean('IrisScleraContrast')
        self.IrisPupilContrast = self['QualityMetrics'].getboolean('IrisPupilContrast')
        self.PupilBoundaryCircularity = self['QualityMetrics'].getboolean('PupilBoundaryCircularity')
        self.GreyScaleUtilization = self['QualityMetrics'].getboolean('GreyScaleUtilization')
        self.IrisRadius = self['QualityMetrics'].getboolean('IrisRadius')
        self.PupilDilation = self['QualityMetrics'].getboolean('PupilDilation')
        self.IrisPupilConcentricity = self['QualityMetrics'].getboolean('IrisPupilConcentricity')
        self.MarginAdequacy = self['QualityMetrics'].getboolean('MarginAdequacy')
        self.Sharpness = self['QualityMetrics'].getboolean('Sharpness')

        with open(self.image_list, "r") as f:
            self.images = f.read().splitlines()
        
        self.image_names = []
        self.image_extensions = []
        
        for image in self.images:
            self.image_names.append(pathlib.Path(image).stem)
            self.image_extensions.append(pathlib.Path(image).suffix)