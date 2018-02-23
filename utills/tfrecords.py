import cv2
from data_loader.data import Data

class TfrecordsUtils(object):

    def __init__(self, config):
        self.config = config
        self.data = Data.getInstance(config)

    def CreateTfrecords(self):
        pass