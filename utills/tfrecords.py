import cv2
import numpy as np
import tensorflow as tf
import sys
import os
from glob import glob
from PIL import Image
from data_loader.data import Data
from data_loader.data import CATEGORY

class TfrecordsUtils(object):

    def __init__(self, Data):
        self.data = Data
        self.config = Data.config

    def load_image(self, filepath):
        # Read, resize and convert to RGB (since cv2 loads images as BGR)
        img = Image.open(filepath)
        img = cv2.imread(filepath)
        img = cv2.resize(img, (self.config["IMAGE_SIZE"], self.config["IMAGE_SIZE"]), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        return img

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def CreateTfRecordsHelper(self, files, labels, type):

        outFile = os.path.abspath(os.path.join(os.getcwd(), "./../tfrecords/" + type + "/" + type + '.tfrecords'))
        try:
            os.remove(outFile)
        except OSError:
            pass

        writer = tf.python_io.TFRecordWriter(outFile)

        for i in range(len(files)):
            # Load image and its label
            img = self.load_image(files[i])
            label = labels[i]

            # Create a feature
            feature = {type + '/label': self._int64_feature(label),
                       type + '/image': self._bytes_feature(tf.compat.as_bytes(img.tobytes()))}

            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write to file
            writer.write(example.SerializeToString())

        writer.close()
        sys.stdout.flush()

    def CreateTfrecords(self):
        (numFiles, train_img, train_labels, val_img,
         val_labels, test_img, test_labels) = self.data.collect_split_data(CATEGORY.paper, CATEGORY.metal, CATEGORY.cardoboard)
        # Create train records
        self.CreateTfRecordsHelper(train_img, train_labels, 'train')
        # Create test records
        self.CreateTfRecordsHelper(test_img, test_labels, 'test')
        # Create validation records
        self.CreateTfRecordsHelper(val_img, val_labels, 'validation')

    def read_from_tfrecords(self, type):
        # TODO: Use glob if there isn't only going to be one file
        file = glob(os.path.abspath(os.path.join(os.getcwd(), "./../tfrecords/" + type + "/" + type + '.tfrecords')))

        feature = {type + '/image': tf.FixedLenFeature([], tf.string),
                   type + '/label': tf.FixedLenFeature([], tf.int64)}

        # Enqueue train.tfrecords
        filename_queue = tf.train.string_input_producer(file, num_epochs=None)

        # Define reader and read file from queue
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)

        # Convert serialized data back to arrays and numbers
        image = tf.decode_raw(features[type + '/image'], tf.float32)
        label = tf.cast(features[type + '/label'], tf.int32)

        # Reshape image data to original shape
        image = tf.reshape(image, [self.config["IMAGE_SIZE"], self.config["IMAGE_SIZE"], 3])

        image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                          batch_size=self.config["BATCH_SIZE"], capacity=10,
                                                          num_threads=2, min_after_dequeue=2)

        return image_batch, label_batch