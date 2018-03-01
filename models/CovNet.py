from base.base_model import BaseModel
import tensorflow as tf
from utills.tfrecords import TfrecordsUtils

class CovNetModel(BaseModel):

    def __init__(self, config):
        # Initialize the base class
        super().__init__(config)
        tfRecords = TfrecordsUtils()
        self.image, self.label = tfRecords.read_from_tfrecords()
        self.build_model()
        self.init_saver()

    def _create_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape=shape,stddev=0.03, dtype=tf.float32))

    def _create_bias(self, shape):
        return tf.Variable(tf.constant(1., shape=shape, dtype=tf.float32))

    def _create_conv2d(self, x, W, _strides):
        return tf.nn.conv2d(input=x,
                            filter=W,
                            strides=_strides,
                            padding='SAME')

    def _create_max_pool_2x2(self, input, _ksize, _strides):
        return tf.nn.max_pool(value=input,
                              ksize=_ksize,
                              strides=_strides,
                              padding='SAME')

    def build_model(self):

        with tf.variable_scope('layer1') as scope:
            filter = self._create_weights([11, 11, 3, 96])
            bias = self._create_bias([96])
            conv = self._create_conv2d(self.image,
                                                filter,
                                                strides = [1,4,4,1])
            preactivation = tf.nn.bias_add(conv, bias)
            conv2d_layer1 = tf.nn.relu(preactivation, name=scope.name)

        # Pool 1
        pool_1 = self._create_max_pool_2x2(conv2d_layer1,
                                                      _ksize=[1,3,3,1],
                                                      _strides= [1,2,2,1])
        with tf.variable_scope('layer2') as scope:
            filter = self._create_weights([5, 5, 96, 192])
            bias = self._create_bias([192])
            conv = self._create_conv2d(pool_1,filter,
                                       strides = [1,1,1,1])
            preactivation = tf.nn.bias_add(conv, bias)
            conv2d_layer2 = tf.nn.relu(preactivation, name=scope.name)

        # Pool 2
        pool_2 = self._create_max_pool_2x2(conv2d_layer2,
                                                      _ksize=[1,3,3,1],
                                                      _strides= [1,2,2,1])
        with tf.variable_scope('layer3') as scope:
            filter = self._create_weights([3, 3, 192, 288])
            bias = self._create_bias([288])
            conv = self._create_conv2d(pool_2,filter,
                                       strides = [1,1,1,1])
            preactivation = tf.nn.bias_add(conv, bias)
            conv2d_layer3 = tf.nn.relu(preactivation, name=scope.name)

        with tf.variable_scope('layer4') as scope:
            filter = self._create_weights([3, 3, 288, 288])
            bias = self._create_bias([288])
            conv = self._create_conv2d(conv2d_layer3,filter,
                                       strides = [1,1,1,1])
            preactivation = tf.nn.bias_add(conv, bias)
            conv2d_layer4 = tf.nn.relu(preactivation, name=scope.name)

        with tf.variable_scope('layer5') as scope:
            filter = self._create_weights([3, 3, 288, 192])
            bias = self._create_bias([288])
            conv = self._create_conv2d(conv2d_layer4,filter,
                                       strides = [1,1,1,1])
            preactivation = tf.nn.bias_add(conv, bias)
            conv2d_layer5 = tf.nn.relu(preactivation, name=scope.name)

        flattened_layer = tf.reshape(conv2d_layer5, [-1, 8 * 8 * 192])

        # First fully connected layer
        with tf.variable_scope('fully_connected_1') as scope:
            weight = tf.Variable(tf.truncated_normal([8 * 8 * 192, 4096], stddev=0.03))
            bias = tf.Variable(tf.truncated_normal([4096], stddev=0.01))
            layer = tf.matmul(flattened_layer, weight)
            fc1 = tf.nn.bias_add(layer, bias)


        # Second fully connected layer
        with tf.variable_scope('fully_connected_2') as scope:
            weight = tf.Variable(tf.truncated_normal([8 * 8 * 192, 4096], stddev=0.03))
            bias = tf.Variable(tf.truncated_normal([4096], stddev=0.01))
            layer = tf.matmul(fc1, weight)
            fc2 = tf.nn.bias_add(layer, bias)

        # Third fully connected layer
        with tf.variable_scope('fully_connected_2') as scope:
            weight = tf.Variable(tf.truncated_normal([8 * 8 * 192, 4096], stddev=0.03))
            bias = tf.Variable(tf.truncated_normal([4096], stddev=0.01))
            layer = tf.matmul(fc2, weight)
            fc3 = tf.nn.bias_add(layer, bias)

        self.prediction = tf.nn.softmax(fc3)




