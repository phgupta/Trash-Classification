# Add get_test_size function in train_trashnet and import that file here
# Use the test_size to go through the test_data only once!

import cv2      # Temp
import glob
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import timedelta
from operator import itemgetter
from sklearn.metrics import confusion_matrix


################### Constants ###################
model_file_name = "Trashnet_FC4_" + str(1e-5)
NUM_EPOCHS = 100

MODEL_PATH = './Model_Info/' + model_file_name + '/model'
CLASS_LABELS = ['cardboard', 'metal', 'paper', 'plastic', 'glass', 'trash']
NUM_CLASSES = len(CLASS_LABELS)

BATCH_SIZE = 1  # CHANGED
NUM_CHANNELS = 3 
IMG_HEIGHT = 512
IMG_WIDTH = 384
IMG_SIZE_FLAT = IMG_HEIGHT * IMG_WIDTH
IMG_SHAPE_LIST = [IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS]
CAPACITY = 100
NUM_THREADS = 4
MIN_AFTER_DEQUEUE = 10


################### Read data ###################
def read_from_tfrecords(fname):

    file = glob.glob(fname+'.tfrecords')
    feature = { fname+'/image': tf.FixedLenFeature([], tf.string),
                fname+'/label': tf.FixedLenFeature([], tf.int64) }

    # Enqueue train.tfrecords
    # 'num_epochs=None' ensures that tf.train.shuffle_batch() can be called indefinitely
    filename_queue = tf.train.string_input_producer(file, num_epochs=None)

    # Define reader and read file from queue
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)

    # Convert serialized data back to arrays and numbers
    image = tf.decode_raw(features[fname+'/image'], tf.float32)
    label = tf.cast(features[fname+'/label'], tf.int32)

    # Reshape image data to original shape
    image = tf.reshape(image, IMG_SHAPE_LIST)

    return image, label

def next_batch(fname):

    # CHECK: Need to compute image, label everytime?
    image, label = read_from_tfrecords(fname)
    image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=BATCH_SIZE, capacity=CAPACITY, 
                                                num_threads=NUM_THREADS, min_after_dequeue=MIN_AFTER_DEQUEUE, 
                                                allow_smaller_final_batch=True)
    label_batch = tf.one_hot(label_batch, NUM_CLASSES)

    return image_batch, label_batch


################### Model ###################
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.5))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input, num_input_channels, filter_size, num_filters, 
                    conv_strides, max_pool_ksize=None, max_pool_strides=None, use_pooling=True):
    
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input, filter=weights, strides=conv_strides, padding='SAME')
    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer, ksize=max_pool_ksize, strides=max_pool_strides, padding='SAME')

    layer = tf.nn.relu(layer)

    return layer, weights

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = np.array(layer_shape[1:4], dtype=int).prod()
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features

def new_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


def model():

    # Convolution Layers
    layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=NUM_CHANNELS,
                                    filter_size=11, num_filters=96, conv_strides=[1,4,4,1],
                                    max_pool_ksize=[1,3,3,1], max_pool_strides=[1,2,2,1], use_pooling=True)

    # print("Type is: ", type(layer_conv1))
    # img = layer_conv1[:, :, 3, 0]
    # print(img)

    
    layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1, num_input_channels=96,
                                    filter_size=5, num_filters=192, conv_strides=[1,1,1,1],
                                    max_pool_ksize=[1,3,3,1], max_pool_strides=[1,2,2,1], use_pooling=True)

    layer_conv3, weights_conv3 = new_conv_layer(input=layer_conv2, num_input_channels=192,
                                    filter_size=3, num_filters=288, conv_strides=[1,1,1,1], use_pooling=False)

    layer_conv4, weights_conv4 = new_conv_layer(input=layer_conv3, num_input_channels=288,
                                    filter_size=3, num_filters=288, conv_strides=[1,1,1,1], use_pooling=False)

    layer_conv5, weights_conv5 = new_conv_layer(input=layer_conv4, num_input_channels=288,
                                    filter_size=3, num_filters=192, conv_strides=[1,1,1,1],
                                    max_pool_ksize=[1,3,3,1], max_pool_strides=[1,2,2,1], use_pooling=True)

    # Fully Connected Layers
    layer_flat, num_features = flatten_layer(layer_conv5)

    layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=4096, use_relu=False)
    layer_fc2 = new_fc_layer(input=layer_fc1, num_inputs=4096, num_outputs=4096, use_relu=False)
    layer_fc3 = new_fc_layer(input=layer_fc2, num_inputs=4096, num_outputs=4096, use_relu=False)
    layer_fc4 = new_fc_layer(input=layer_fc3, num_inputs=4096, num_outputs=4096, use_relu=False)
    layer_fc5 = new_fc_layer(input=layer_fc4, num_inputs=4096, num_outputs=NUM_CLASSES, use_relu=False)

    return layer_fc5


def calc_metrics(matrix):
    # Diagonals of confusion matrix
    diag = np.diagonal(matrix)

    # True positives: Diagonals of confusion matrix
    # tp = dict(enumerate(diag))
    tp = diag

    # False positives: Sum of corresponding column values (excluding TP)
    col_sum = np.sum(matrix, axis=0)
    # fp = dict(enumerate(col_sum - diag))
    fp = col_sum - diag

    # False negatives: Sum of corresponding row values (excluding TP)
    row_sum = np.sum(matrix, axis=1)
    # fn = dict(enumerate(row_sum - diag))
    fn = row_sum - diag

    # True negatives: Sum of rows & cols, excluding that class's rows & cols
    matrix_sum = sum(row_sum)
    # tn = dict(enumerate([matrix_sum - row_sum[i] - col_sum[i] + diag[i] for i in range(len(matrix))]))
    tn = matrix_sum - row_sum - col_sum + diag

    return tp, fp, tn, fn


def roc_curve(tp, fp, tn, fn):
    
    color_list = ['brown', 'green', 'yellow', 'black', 'pink', 'purple']
    plt.figure(figsize=(10,10))
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)

    for j in range(NUM_CLASSES):
        tpr = [(tp[i][j] / (tp[i][j] + fn[i][j])) for i in range(NUM_EPOCHS)]
        fpr = [(tn[i][j] / (tn[i][j] + fp[i][j])) for i in range(NUM_EPOCHS)]

        tpr.sort()
        fpr.sort()

        plt.plot(fpr0, tpr0, color=color_list[j], label='class'+str(j), linewidth=2)
    
    plt.savefig('roc_curve')

    # # Class 0 only
    # tpr0 = [(tp[i][0] / (tp[i][0] + fn[i][0])) for i in range(NUM_EPOCHS)]
    # fpr0 = [(tn[i][0] / (tn[i][0] + fp[i][0])) for i in range(NUM_EPOCHS)]

    # # CHECK: The two lists are related!
    # # c = list(zip(tpr0, fpr0))
    # # c = sorted(c, key=itemgetter(0))
    # # tpr0, fpr0 = zip(*c)

    # tpr0.sort()
    # fpr0.sort()

    # plt.figure(figsize=(10,10))
    # plt.xlabel("FPR", fontsize=14)
    # plt.ylabel("TPR", fontsize=14)
    # plt.title("ROC Curve", fontsize=14)

    # plt.plot(fpr0, tpr0, color='green', linewidth=2)
    # # plt.plot([0,1], [0,1], color='navy', linewidth=2, linestyle='--')
    # plt.savefig('roc_curve')


def pr_curve(tp, fp, tn, fn):
    # Class 0 only
    precision = [(tp[i][0] / (tp[i][0] + fp[i][0])) for i in range(NUM_EPOCHS)]
    recall = [(tp[i][0] / (tp[i][0] + fn[i][0])) for i in range(NUM_EPOCHS)]
    
    # precision.sort()
    # recall.sort()

    plt.figure(figsize=(10,10))
    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.title("PR Curve", fontsize=14)

    plt.plot(recall, precision, color='green', linewidth=2)
    plt.savefig('pr_curve')


################### Placeholders ###################
x = tf.placeholder(tf.float32, shape=[None, IMG_SIZE_FLAT], name='x')
x_image = tf.reshape(x, [-1, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS])
y_true = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)


################### Get next batch of data ###################
test_img_batch, test_lbl_batch = next_batch('test')
model_output = model()


################### Result ###################
y_pred = tf.nn.softmax(model_output)
y_pred_cls = tf.argmax(y_pred, axis=1)


################### Performance Measure ###################
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


################### Save & Restore Model ###################
saver = tf.train.Saver()


################### Main Function ###################
def test_accuracy():
    tp = np.zeros([NUM_EPOCHS, NUM_CLASSES])
    fp = np.zeros([NUM_EPOCHS, NUM_CLASSES])
    tn = np.zeros([NUM_EPOCHS, NUM_CLASSES])
    fn = np.zeros([NUM_EPOCHS, NUM_CLASSES])

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    start_time = time.time()    

    for i in range(NUM_EPOCHS):
        x_test_batch, y_test_batch = sess.run([test_img_batch, test_lbl_batch])
        feed_dict_test = {
            x_image: x_test_batch,
            y_true: y_test_batch
        }
        cls_y_pred, cls_y_true = sess.run([y_pred_cls, y_true_cls], feed_dict=feed_dict_test)

        feed_dict_test_acc = {
            y_pred_cls: cls_y_pred,
            y_true_cls: cls_y_true
        }
        test_accuracy = sess.run(accuracy, feed_dict=feed_dict_test_acc)

        # Print to stdout & file
        # print_epoch_accuracy = "Epoch: " + str(i) + "\tAccuracy: {0:.1%}"
        # print(print_epoch_accuracy.format(test_accuracy))

        image = sess.run(layer_conv1)
        # print("Type is: ", type(layer_conv1))
        img = image[:, :, 3, 0]
        print(img)

        break
        # f.write(print_epoch_accuracy.format(test_accuracy)+"\n")

        # # Calculate tp, fp, tn, fn
        # matrix = confusion_matrix(cls_y_true, cls_y_pred)
        # print(matrix)
        # tp[i], fp[i], tn[i], fn[i] = calc_metrics(matrix)

    # # Plot roc_curve & pr_curve
    # roc_curve(tp, fp, tn, fn)
    # pr_curve(tp, fp, tn, fn)
    
    coord.request_stop()
    coord.join(threads)

    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    # f.write("Time usage: " + str(timedelta(seconds=int(round(time_dif))))+"\n")


################### Sessions ###################
with tf.Session() as sess:

    # Open file for logging performance metrics
    # f = open(model_file_name + '_test.txt', 'a+')

    # Initialize global variables
    sess.run(tf.global_variables_initializer())

    # Restore model weights from previously saved model
    saver.restore(sess, MODEL_PATH)
    print("Model restored from file: %s" % MODEL_PATH)

    # Main function: Test CNN on testing data
    test_accuracy()   

    # Close file
    # f.close()

################### Comments ###################
# 1.
# To see the images in the different batches,
# Add below code right after 'x_batch, y_true_batch = sess.run([img_batch, lbl_batch])'
# Change NUM_EPOCHS to 3 and create folders - "Images", "Images/Batch0", ...
# from PIL import Image
# for j in range(TRAIN_BATCH_SIZE):
#     img = Image.fromarray(x_batch[j], 'RGB')
#     img.save('Images/Batch' + str(i) + '/' + str(j) + '.png')