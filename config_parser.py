import configparser
import pathlib

#config = configparser.ConfigParser()

class Config(configparser.ConfigParser):
    
    def __init__(self, config_file):
        super().__init__()
        
        self.read(config_file)
        self.image_list = self['Input']['image_list_path']
        self.image_path = self['Input']['image_directory_path']
        self.parameter_path = self['Input']['parameters_path']
        self.parameters_suffix = self['Input']['parameters_suffix']
        self.mask_path = self['Input']['mask_path']
        self.convert_from_OSIRIS = self['Input'].getboolean('OSIRIS_formatting')
        self.mask_suffix = self['Input']['mask_suffix']
        
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
        
        self.check_thresholds = self['Output'].getboolean('check_thresholds')
        self.output_to_file = self['Output'].getboolean('output_to_file')
        self.output_to_console = self['Output'].getboolean('output_to_CLI')
        self.output_file_path = self['Output']['output_file_path']
        self.round_to = self['Output'].getint('round_to')
        
        self.thresh_uia = self['Thresholds']['UsableIrisArea'].split()
        self.thresh_isc = self['Thresholds']['IrisScleraContrast'].split()
        self.thresh_ipc = self['Thresholds']['IrisPupilContrast'].split()
        self.thresh_pbc = self['Thresholds']['PupilBoundaryCircularity'].split()
        self.thresh_gsu = self['Thresholds']['GreyScaleUtilization'].split()
        self.thresh_ir = self['Thresholds']['IrisRadius'].split()
        self.thresh_pd = self['Thresholds']['PupilDilation'].split()
        self.thresh_ipcon = self['Thresholds']['IrisPupilConcentricity'].split()
        self.thresh_ma = self['Thresholds']['MarginAdequacy'].split()
        self.thresh_sh = self['Thresholds']['Sharpness'].split()
        
        self.param_table = self.set_param_table()
            
        self.__reformat_thresholds()

        try:
            with open(self.image_list, "r") as f:
                self.images = f.read().splitlines()
        except FileNotFoundError:
            print("Image list file not found.")
            exit()
        
        self.image_names = []
        self.image_extensions = []
        
        for image in self.images:
            self.image_names.append(pathlib.Path(image).stem)
            self.image_extensions.append(pathlib.Path(image).suffix)
            
            
    def __reformat_thresholds(self):
        for thresh in [self.thresh_uia, self.thresh_isc, self.thresh_ipc,
                       self.thresh_pbc, self.thresh_gsu, self.thresh_ir,
                       self.thresh_pd, self.thresh_ipcon, self.thresh_ma, self.thresh_sh]:
            for i in range(len(thresh)):
                try:
                    if str(thresh[i]) == 'None':
                        thresh[i] = None
                    else:
                        thresh[i] = float(thresh[i])
                except ValueError:
                    print("Thresholds must be floats or None. Problematic value: {}".format(thresh[i]))
                    exit()
                except TypeError:
                    print("Thresholds must be floats or None. Problematic value: {}".format(thresh[i]))
                    exit()
                    
        
    def set_param_table(self):
        param_table = []
        for param in self['QualityMetrics']:
            if self['QualityMetrics'][param] != 'False':
                param_table.append(param)
        return param_table
    