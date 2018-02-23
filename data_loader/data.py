import glob
import os
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from utills.config import process_config
import itertools
from enum import Enum

class CATEGORY(Enum):
    paper = 0
    metal = 1
    cardoboard = 2
    trash = 3
    glass = 4
    plastic = 5

class Data(object):

    __instance = None
    image_list = []
    test = None
    train = None

    @staticmethod
    def getInstance(_config):
        if Data.__instance == None:
            Data(config=_config)
        return Data.__instance

    def __init__(self, config):
        if Data.__instance != None:
            raise AttributeError
        else:
            self.config = config
            Data.__instance = self

    def getLabeledImages(self):
        path = self.config.PATH_TO_DATA
        for root, dirs, files in os.walk(path, topdown=True):
            for directory in dirs:
                for filename in glob.glob(os.path.join(path,directory) + '/*.jpg'):
                    imgObj = {}
                    if directory == "paper":
                        imgObj['data'] = np.array(Image.open(filename))
                        imgObj['label'] = 0
                    elif directory == "metal":
                        imgObj['data'] = np.array(Image.open(filename))
                        imgObj['label'] = 1
                    elif directory == "cardboard":
                        imgObj['data'] = np.array(Image.open(filename))
                        imgObj['label'] = 2
                    elif directory == "trash":
                        imgObj['data'] = np.array(Image.open(filename))
                        imgObj['label'] = 3
                    elif directory == "glass":
                        imgObj['data'] = np.array(Image.open(filename))
                        imgObj['label'] = 4
                    elif directory == "plastic":
                        imgObj['data'] = np.array(Image.open(filename))
                        imgObj['label'] = 5
                    self.image_list.append(imgObj)

    # Return a list of images from the label
    def getImagesByLabel(self, *labels):
        retList = list(map((lambda l: list(filter(lambda img: img['label'] == l, self.image_list))), labels))
        return list(itertools.chain.from_iterable(retList))

    # Get the files from PAPER, METAL and CARDBOARD or list of lables
    # Used in creating Tf records
    def collect_split_data(self, *labels):
        _labels = []
        def files(_category):
            _files = glob.glob(os.path.join(self.config.PATH_TO_DATA, _category.name+ "/*.jpg"))
            _labels = list(map(lambda x: _category.value , [None] * len(_files)))
            return list(zip(_files, _labels))
        fileList = list(map(lambda category : files(category),labels))
        files, labels = zip(*list(itertools.chain.from_iterable(fileList)))
        train_img, test_img, train_labels, test_labels = train_test_split(files,
                labels, test_size= self.config.TEST_SIZE)
        train_img, val_img, train_labels, val_labels = train_test_split(train_img,
                train_labels, test_size= self.config.VAL_SIZE)
        return (len(files), train_img, train_labels, val_img, val_labels, test_img, test_labels)



    # Function to get equal distribution of training and testing from each category
    def splitData(self):
        lTrain = []
        lTest = []
        if(self.image_list):
            imgdata = pd.DataFrame(self.image_list)
            groupedByLabel = imgdata.groupby(['label'])
            for group in groupedByLabel.groups:
                _train, _test = train_test_split(groupedByLabel.get_group(group), test_size=0.2)
                lTrain.append(_train.values)
                lTest.append(_test.values)
            self.train = np.vstack(np.array(lTrain))
            self.test = np.vstack(np.array(lTest))
        else:
            raise Exception("Images aren't processed yet. Call function getLabeledImages()")

    # Will return a list of indexes for every fold
    def getCrossFolds(self, _seed):
        train = []
        test = []
        if(self.image_list):
            imgdata = pd.DataFrame(self.image_list)
            _data, _label = np.array_split(imgdata.values, 2, axis=1)
            data = _data.flatten()
            label = _label.flatten()
            cv = KFold(n_splits=10, random_state=_seed)
            for index_train, index_test in cv.split(data):
                train.append(index_train)
                test.append(index_test)
            return train, test

def main():
    # Parse the config file
    config = process_config("././configs/config.json")
    assert os.path.isdir(config.PATH_TO_DATA)
    # Only keeping one instance of data
    data = Data.getInstance(config)
    files = data.collect_split_data(CATEGORY.paper, CATEGORY.metal)
    # data.getLabeledImages()
    # labledImages = data.getImagesByLabel(0,1,2)
    # data.splitData()
    # data.getCrossFolds()

    return 0

if __name__ == "__main__":
    main()