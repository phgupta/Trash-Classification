import glob
import cv2
import sys
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from random import shuffle
from datetime import timedelta


################### Constants ###################
DATASET_PATH = "Dataset/**/*.jpg"
TRAIN_SIZE = 0.8
TEST_SIZE = 0.2

CLASS_LABELS = ['cardboard', 'metal', 'paper']
NUM_CLASSES = len(CLASS_LABELS)
NUM_CHANNELS = 3 

IMG_SIZE = 256
IMG_SHAPE = (IMG_SIZE, IMG_SIZE)


################### Collect & Split Data ###################
def collect_split_data():
    print("collect_split_data()")

    labels = []
    files = glob.glob(DATASET_PATH)

    for file in files:
        if CLASS_LABELS[0] in file:
            labels.append(0)
        elif CLASS_LABELS[1] in file:
            labels.append(1)
        elif CLASS_LABELS[2] in file:
            labels.append(2)
        else:
            print("Error: Image filename does not contain correct label.")
             
    c = list(zip(files, labels))
    shuffle(c)
    files, labels = zip(*c)

    train_img = files[0:int(TRAIN_SIZE * len(files))]
    train_labels = labels[0:int(TRAIN_SIZE * len(files))]
    test_img = files[int(TRAIN_SIZE * len(files)):]
    test_labels = labels[int(TRAIN_SIZE * len(files)):]

    return train_img, train_labels, test_img, test_labels


################### Write to .tfrecords ###################
def load_image(addr):
    # Read, resize and convert to RGB (since cv2 loads images as BGR)
    img = Image.open(addr)
    img = cv2.imread(addr)
    img = cv2.resize(img, IMG_SHAPE, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_tfrecord(files, labels, fname):
    print("create_tfrecord(" + fname + ")")
    start_time = time.time()

    # Open .tfrecords file
    writer = tf.python_io.TFRecordWriter(fname + '.tfrecords')
    
    for i in range(len(files)):
        # Load image and its label
        img = load_image(files[i])
        label = labels[i]

        # Create a feature
        feature = { fname+'/label': _int64_feature(label),
                    fname+'/image': _bytes_feature(tf.compat.as_bytes(img.tobytes())) }

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write to file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()

    end_time = time.time()
    time_diff = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_diff)))))


def create_tfrecords(train_img, train_labels, test_img, test_labels):
    print("create_tfrecords()")
    create_tfrecord(train_img, train_labels, 'train')
    create_tfrecord(test_img, test_labels, 'test')


 ################### Main ###################
train_img, train_labels, test_img, test_labels = collect_split_data()
create_tfrecords(train_img, train_labels, test_img, test_labels)
