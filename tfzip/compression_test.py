# Code for TensorFlow model compression based on pruning
# from "Learning both Weights and Connections for Efficient Neural Networks"
# by Han et. al 2015
import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
TEST_KEEP_PROB = 1.0


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# LeNet-300-100 for MNIST; expect ~1.6% error / 98.4% accuracy
# LeNet 300-100 has 267K; should reduce to 22K
# Base parameter count is 267K = (28 * 28 * 300 + 300 * 100 + 100 * 10) + (300 + 100 + 10)
threshold = 0.0
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
# fc1
W_fc1 = weight_variable([28 * 28, 300])
b_fc1 = bias_variable([300])
h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
# Dropout
keep_prob1 = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob1)
# fc2
W_fc2 = weight_variable([300, 100])
b_fc2 = bias_variable([100])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# Dropout
keep_prob2 = tf.placeholder("float")
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob2)
# fc3
W_fc3 = weight_variable([100, 10])
b_fc3 = bias_variable([10])
logits = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
y = tf.nn.softmax(logits)

# Define loss function, optimization technique, and accuracy metric
# Add epsilon to prevent 0 log 0; See http://quabr.com/33712178/tensorflow-nan-bug
# http://stackoverflow.com/questions/34240703/difference-between-tensorflow-tf-nn-softmax-and-tf-nn-softmax-cross-entropy-with
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_)
l2reg = tf.nn.l2_loss(W_fc3)
loss = cross_entropy + l2reg
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# Define pruning ops
threshold = 0.0  # Placeholder later
indicator_matrix1 = tf.to_float(tf.greater_equal(W_fc1, tf.constant(threshold, shape=W_fc1.get_shape())))
indicator_matrix2 = tf.to_float(tf.greater_equal(W_fc2, tf.constant(threshold, shape=W_fc2.get_shape())))
indicator_matrix3 = tf.to_float(tf.greater_equal(W_fc3, tf.constant(threshold, shape=W_fc3.get_shape())))
fc1_pruned = tf.mul(W_fc1, indicator_matrix1)
fc2_pruned = tf.mul(W_fc2, indicator_matrix2)
fc3_pruned = tf.mul(W_fc3, indicator_matrix3)
prune_fc1 = W_fc1.assign(fc1_pruned)
prune_fc2 = W_fc2.assign(fc2_pruned)
prune_fc3 = W_fc3.assign(fc3_pruned)
prune_all = tf.group(prune_fc1, prune_fc2, prune_fc3)

# Helper ops
nonzero_indicator1 = tf.to_float(tf.not_equal(W_fc1, tf.constant(0.0, shape=W_fc1.get_shape())))
nonzero_indicator2 = tf.to_float(tf.not_equal(W_fc2, tf.constant(0.0, shape=W_fc2.get_shape())))
nonzero_indicator3 = tf.to_float(tf.not_equal(W_fc3, tf.constant(0.0, shape=W_fc3.get_shape())))
count_parameters1 = tf.reduce_sum(nonzero_indicator1)
count_parameters2 = tf.reduce_sum(nonzero_indicator2)
count_parameters3 = tf.reduce_sum(nonzero_indicator3)

# Create a saver for writing training checkpoints.
saver = tf.train.Saver()
# Run training in a session
sess = tf.InteractiveSession()


def train(iterations=20000, kp1=0.5, kp2=0.5):
    for i in range(iterations):
        batch_xs, batch_ys = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys, keep_prob1: TEST_KEEP_PROB, keep_prob2: TEST_KEEP_PROB})
            print("step %d training accuracy %g" % (i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob1: kp1, keep_prob2: kp2})
    print("test accuracy %g" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob1: TEST_KEEP_PROB, keep_prob2: TEST_KEEP_PROB}))


def print_parameter_counts():
    print("Parameters1: {0}".format(sess.run(count_parameters1)))
    print("Parameters2: {0}".format(sess.run(count_parameters2)))
    print("Parameters3: {0}".format(sess.run(count_parameters3)))


def calculate_new_keep_prob(original_keep_prob, original_connections, retraining_connections):
    return 1.0 - ((1.0 - original_keep_prob) * math.sqrt(retraining_connections / original_connections))


def compress(times=1):
    kp1 = kp2 = 0.5
    for i in range(times):
        print("# Compressing {0}".format(i))
        c1 = sess.run(count_parameters1)
        c2 = sess.run(count_parameters2)
        sess.run(prune_all)
        c1_retrain = sess.run(count_parameters1)
        c2_retrain = sess.run(count_parameters2)
        kp1 = calculate_new_keep_prob(kp1, c1, c1_retrain)
        kp2 = calculate_new_keep_prob(kp2, c2, c2_retrain)
        # Retrain with pruned connections
        print("# Before retraining")
        print_parameter_counts()
        train(20000, kp1, kp2)
        print("# After retraining")
        print_parameter_counts()
    sess.run(prune_all)
    saver.save(sess, "compressed_model/compressed_model")
    print("# Saved compressed model")
    print_parameter_counts()


if __name__ == '__main__':
    # Restore a model from a checkpoint or train if missing
    checkpoint = tf.train.get_checkpoint_state("uncompressed_model")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("test accuracy %g" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob1: TEST_KEEP_PROB, keep_prob2: TEST_KEEP_PROB}))
    else:
        sess.run(tf.initialize_all_variables())
        train()     # test accuracy 0.9747
        saver.save(sess, "uncompressed_model/uncompressed_model")

    # Run compression on the result
    compress(5)
