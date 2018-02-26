import os
import sys
from data_loader.data import Data
from data_loader.data import CATEGORY
from utills.config import process_config
from utills.tfrecords import TfrecordsUtils

def main():
    # Parse the config file
    config = process_config("./../configs/config.json")
    assert os.path.isdir(config.PATH_TO_DATA)
    # Only keeping one instance of data
    data = Data.getInstance(config)
    # data.getLabeledImages()
    # labledImages = data.getImagesByLabel(0,1,2)
    # data.splitData()
    # data.getCrossFolds()
    tfRecordUtil = TfrecordsUtils(data)
    tfRecordUtil.CreateTfrecords()
    image, label = tfRecordUtil.read_from_tfrecords('train')
    return 0

if __name__ == "__main__":
    main()