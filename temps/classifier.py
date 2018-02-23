import numpy as np
import glob
import keras
import tensorflow as tf
from sklearn.model_selection import KFold

def classifier(train_dataset, test_dataset, train_labels, test_labels, trainsize, testsize):
    batch_size = 16
    patch_size = 5
    image_size1 = 384
    image_size2 = 512
    depth = 16
    num_hidden1 = 256
    num_hidden2 = 64
    num_channels = 3
    num_labels = 4

    train_dataset = train_dataset.reshape(
            (trainsize, image_size1, image_size2, num_channels)).astype(np.float32)
    test_dataset = test_dataset.reshape(
            (testsize, image_size1, image_size2, num_channels)).astype(np.float32)

    def accuracy(predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

    graph = tf.Graph()

    with graph.as_default():
        # Define the training dataset and lables
        tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size1, image_size2, num_channels))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

        # Validation/test dataset
        tf_test_dataset = tf.constant(test_dataset)

        # CNN layer 1 with filter (num_channels, depth) (3, 16)
        cnn1_W = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, num_channels, depth], stddev=0.1))
        cnn1_b = tf.Variable(tf.zeros([depth]))

        # CNN layer 2 with filter (depth, depth) (16, 16)
        cnn2_W = tf.Variable(tf.truncated_normal(
            [patch_size, patch_size, depth, depth], stddev=0.1))
        cnn2_b = tf.Variable(tf.constant(1.0, shape=[depth]))

        # Compute the output size of the CNN2 as a 1D array.
        size = image_size1 // 4 * image_size2 // 4 * depth

        # FC1 (size, num_hidden1) (size, 256)
        fc1_W = tf.Variable(tf.truncated_normal(
            [size, num_hidden1], stddev=np.sqrt(2.0 / size)))
        fc1_b = tf.Variable(tf.constant(1.0, shape=[num_hidden1]))

        # FC2 (num_hidden1, num_hidden2) (size, 64)
        fc2_W = tf.Variable(tf.truncated_normal(
            [num_hidden1, num_hidden2], stddev=np.sqrt(2.0 / (num_hidden1))))
        fc2_b = tf.Variable(tf.constant(1.0, shape=[num_hidden2]))

        # Classifier (num_hidden2, num_labels) (64, 10)
        classifier_W = tf.Variable(tf.truncated_normal(
            [num_hidden2, num_labels], stddev=np.sqrt(2.0 / (num_hidden2))))
        classifier_b = tf.Variable(tf.constant(1.0, shape=[num_labels]))

        # Model.
        def model(data):
        # First convolution layer with stride = 1 and pad the edge to make the output size the same.
        # Apply ReLU and a maximum 2x2 pool
            conv1 = tf.nn.conv2d(data, cnn1_W, [1, 1, 1, 1], padding='SAME')
            hidden1 = tf.nn.relu(conv1 + cnn1_b)
            pool1 = tf.nn.max_pool(hidden1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

            # Second convolution layer
            conv2 = tf.nn.conv2d(pool1, cnn2_W, [1, 1, 1, 1], padding='SAME')
            hidden2 = tf.nn.relu(conv2 + cnn2_b)
            pool2 = tf.nn.max_pool(hidden2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

            # Flattern the convolution output
            shape = pool2.get_shape().as_list()
            reshape = tf.reshape(pool2, [shape[0], shape[1] * shape[2] * shape[3]])

            # 2 FC hidden layers
            fc1 = tf.nn.relu(tf.matmul(reshape, fc1_W) + fc1_b)
            fc2 = tf.nn.relu(tf.matmul(fc1, fc2_W) + fc2_b)

            # Return the result of the classifier
            return tf.matmul(fc2, classifier_W) + classifier_b

        # Training computation.
        logits = model(tf_train_dataset)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

        # Optimizer.
        optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        test_prediction = tf.nn.softmax(model(tf_test_dataset))

    num_steps = 20001

    with tf.Session(graph=graph) as session:
      tf.global_variables_initializer().run()
      print('Initialized')
      for step in range(num_steps):
          offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
          batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
          batch_labels = train_labels[offset:(offset + batch_size), :]
          feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
          _, l, predictions = session.run(
              [optimizer, loss, train_prediction], feed_dict=feed_dict)
          if (step % 500 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))