import time
import numpy as np
import tensorflow as tf
from datetime import timedelta
from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets('./MNIST/', one_hot=True)

############ Constants ############
FILTER_SIZE1 = 5
NUM_FILTERS1 = 16
FILTER_SIZE2 = 5
NUM_FILTERS2 = 36
FC_SIZE = 128

IMG_SIZE = 28
IMG_SIZE_FLAT = IMG_SIZE * IMG_SIZE
IMG_SHAPE = (IMG_SIZE, IMG_SIZE)
NUM_CHANNELS = 1
NUM_CLASSES = 10

TRAIN_BATCH_SIZE = 64

############ Functions ############
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

def optimize(num_iterations):
		global total_iterations
		start_time = time.time()

		for i in range(total_iterations, total_iterations+num_iterations):
			x_batch, y_true_batch = data.train.next_batch(TRAIN_BATCH_SIZE)
			feed_dict_train = {
				x: x_batch,
				y_true: y_true_batch
			}
			sess.run(optimizer, feed_dict=feed_dict_train)

			if (i % 100 == 0):
				acc = sess.run(accuracy, feed_dict=feed_dict_train)
				msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
				print(msg.format(i + 1, acc))

		total_iterations += num_iterations
		end_time = time.time()
		time_dif = end_time - start_time
		print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

def model():
	layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=NUM_CHANNELS,
											filter_size=FILTER_SIZE1, num_filters=NUM_FILTERS1, use_pooling=True)
	layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1, num_input_channels=NUM_FILTERS1,
											filter_size=FILTER_SIZE2, num_filters=NUM_FILTERS2, use_pooling=True)

	layer_flat, num_features = flatten_layer(layer_conv2)
	layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=FC_SIZE, use_relu=True)
	layer_fc2 = new_fc_layer(input=layer_fc1, num_inputs=FC_SIZE, num_outputs=NUM_CLASSES, use_relu=False)

	return layer_fc2


############ Placeholders ############
x = tf.placeholder(tf.float32, shape=[None, IMG_SIZE_FLAT], name='x')
x_image = tf.reshape(x, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


############ Optimization ############
model_output = model()
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=model_output, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)


############ Result ############
y_pred = tf.nn.softmax(model_output)
y_pred_cls = tf.argmax(y_pred, dimension=1)


############ Performance Measure ############
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


############ Session ############
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	total_iterations = 0
	optimize(5000)