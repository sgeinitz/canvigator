import os


class CanvigatorConfig:
    """ Canvigator configuration class, mostly for file paths """

    def __init__(self):
        """ Simple file path configuration for data and figures (for both input and output). """
        self.data_path = os.getcwd() + "/data/"
        self.figures_path = os.getcwd() + "/figures/"
        self.quiz_prefix = "quiz_"
