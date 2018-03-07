import glob
import time
import numpy as np
import tensorflow as tf
from datetime import timedelta


################### Constants ###################
CLASS_LABELS = ['cardboard', 'metal', 'paper']
NUM_CLASSES = len(CLASS_LABELS)
NUM_CHANNELS = 3 

IMG_SIZE = 256
IMG_SIZE_FLAT = IMG_SIZE * IMG_SIZE
IMG_SHAPE_LIST = [IMG_SIZE, IMG_SIZE, NUM_CHANNELS]

FILTER_SIZE1 = 5
NUM_FILTERS1 = 16
FILTER_SIZE2 = 5
NUM_FILTERS2 = 36
FC_SIZE = 128

BATCH_SIZE = 64
CAPACITY = 100
NUM_THREADS = 4
MIN_AFTER_DEQUEUE = 10
LEARNING_RATE = 1e-4

NUM_EPOCHS = 100
ACC_COUNT = 10

MODEL_PATH = './Model_Info/model'


################### Get next batch ###################
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

    # CHECK: Need to compute image, label everytime or nah?
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

def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1,1,1,1], padding='SAME')
    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

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
    layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=NUM_CHANNELS,
                                            filter_size=FILTER_SIZE1, num_filters=NUM_FILTERS1, use_pooling=True)
    layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1, num_input_channels=NUM_FILTERS1,
                                            filter_size=FILTER_SIZE2, num_filters=NUM_FILTERS2, use_pooling=True)

    layer_flat, num_features = flatten_layer(layer_conv2)
    layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=FC_SIZE, use_relu=True)
    layer_fc2 = new_fc_layer(input=layer_fc1, num_inputs=FC_SIZE, num_outputs=NUM_CLASSES, use_relu=False)

    return layer_fc2


################### Main Function ###################
def train_model():
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    start_time = time.time()

    for i in range(NUM_EPOCHS):
        print("Iteration: " + str(i))

        x_batch, y_true_batch = sess.run([img_batch, lbl_batch])
        feed_dict_train = {
            x_image: x_batch,
            y_true: y_true_batch
        }

        sess.run(optimizer, feed_dict=feed_dict_train)

        if (i % ACC_COUNT == 0):
            acc = sess.run(accuracy, feed_dict=feed_dict_train)
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i + 1, acc))

    coord.request_stop()
    coord.join(threads)

    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


################### Placeholders ###################
x = tf.placeholder(tf.float32, shape=[None, IMG_SIZE_FLAT], name='x')
x_image = tf.reshape(x, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
y_true = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)


################### Optimization ###################
img_batch, lbl_batch = next_batch('train')
# test_img_batch, test_lbl_batch = next_batch('test')
model_output = model()
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=model_output, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)


################### Result ###################
y_pred = tf.nn.softmax(model_output)
y_pred_cls = tf.argmax(y_pred, axis=1)


################### Performance Measure ###################
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


################### Save & Restore Model ###################
saver = tf.train.Saver()


################### Sessions ###################
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_model()
    
    # Save model weights to disk
    save_path = saver.save(sess, MODEL_PATH)
    print("Model saved in file: %s" % save_path)


################### Comments ###################
# 1.
# To see the images in the different batches,
# Add below code right after 'x_batch, y_true_batch = sess.run([img_batch, lbl_batch])'
# Change NUM_EPOCHS to 3 and create folders - "Images", "Images/Batch0", ...
# from PIL import Image
# for j in range(TRAIN_BATCH_SIZE):
#     img = Image.fromarray(x_batch[j], 'RGB')
#     img.save('Images/Batch' + str(i) + '/' + str(j) + '.png')