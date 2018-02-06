from PIL import Image
import numpy as np
import glob
import keras
from sklearn.model_selection import KFold
from classifier import classifier

def readData():
    image_list = []
    myclass = []
    for filename in glob.glob('dataset-resized/glass/*.jpg'): #assuming gif
        im=np.array(Image.open(filename))
        myclass.append(0)
        image_list.append(im)
    for filename in glob.glob('dataset-resized/cardboard/*.jpg'): #assuming gif
        im=np.array(Image.open(filename))
        image_list.append(im)
        myclass.append(1)
    for filename in glob.glob('dataset-resized/metal/*.jpg'): #assuming gif
        im=np.array(Image.open(filename))
        image_list.append(im)
        myclass.append(2)
    for filename in glob.glob('dataset-resized/paper/*.jpg'): #assuming gif
        im=np.array(Image.open(filename))
        image_list.append(im)
        myclass.append(3)
    
    myclass = keras.utils.to_categorical(np.asarray(myclass), num_classes=4)
    return image_list, myclass


[x,y] = readData()
x = np.asarray(x)
# pca
cv = KFold(n_splits = 10, random_state = 43)
for index_train, index_test in cv.split(x):
    trainsize = index_train.size
    testsize = index_test.size
    train_dataset = x[index_train]
    test_dataset = x[index_test]
    train_labels = y[index_train] 
    test_labels = y[index_test]
    classifier(train_dataset, test_dataset, train_labels, test_labels, trainsize, testsize)
    