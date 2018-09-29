import glob
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.metrics import confusion_matrix


################### Constants ###################
# 1e-3 and 1e-4 are good!
# LEARNING_RATE_LIST = [1e-2, 1e-3, 1e-4, 1e-5]
LEARNING_RATE = 1e-5
model_file_name = "Trashnet_FC4_" + str(LEARNING_RATE)
NUM_EPOCHS = 100
ACC_COUNT = 10

MODEL_PATH = './Model_Info/' + model_file_name + '/model'
CLASS_LABELS = ['cardboard', 'metal', 'paper', 'plastic', 'glass', 'trash']
NUM_CLASSES = len(CLASS_LABELS)

BATCH_SIZE = 32
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


################### Placeholders ###################
x = tf.placeholder(tf.float32, shape=[None, IMG_SIZE_FLAT], name='x')
x_image = tf.reshape(x, [-1, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS])
y_true = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)


################### Get next batch of data ###################
img_batch, lbl_batch = next_batch('train')
model_output = model()


################### Result ###################
y_pred = tf.nn.softmax(model_output)
y_pred_cls = tf.argmax(y_pred, axis=1)


################### Performance Measure ###################
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


################### Optimization ###################
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=model_output, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)


################### Save & Restore Model ###################
saver = tf.train.Saver()


################### Main Function ###################
def train_model():
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    start_time = time.time()

    for i in range(NUM_EPOCHS):
        
        x_batch, y_true_batch = sess.run([img_batch, lbl_batch])
        feed_dict_train = {
            x_image: x_batch,
            y_true: y_true_batch
        }

        _, loss = sess.run([optimizer, cost], feed_dict=feed_dict_train)
        
        print_epoch_loss = "Epoch: " + str(i) + "\tLoss: " + str(loss)
        print(print_epoch_loss)
        f.write(print_epoch_loss+"\n")

        if (i % ACC_COUNT == 0):
            acc = sess.run(accuracy, feed_dict=feed_dict_train)
            
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i + 1, acc))
            f.write(msg.format(i+1, acc)+"\n")

    coord.request_stop()
    coord.join(threads)

    end_time = time.time()
    time_dif = end_time - start_time
    
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    f.write("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


################### Sessions ###################
with tf.Session() as sess:

    # Open file for logging performance metrics
    f = open(model_file_name + '.txt', 'a+')

    # Initialize global variables
    sess.run(tf.global_variables_initializer())

    # Main function: Start training the CNN
    train_model()
    
    # Save model weights to disk
    save_path = saver.save(sess, MODEL_PATH)
    print("Model saved in file: %s" % save_path)

    # Close file
    f.close()


################### Comments ###################
# 1.
# To see the images in the different batches,
# Add below code right after 'x_batch, y_true_batch = sess.run([img_batch, lbl_batch])'
# Change NUM_EPOCHS to 3 and create folders - "Images", "Images/Batch0", ...
# from PIL import Image
# for j in range(TRAIN_BATCH_SIZE):
#     img = Image.fromarray(x_batch[j], 'RGB')
#     img.save('Images/Batch' + str(i) + '/' + str(j) + '.png')