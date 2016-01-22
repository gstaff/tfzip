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
# "Caffe was modified to add a mask which disregards pruned parameters during network operation for each weight tensor"
tf.GraphKeys.PRUNING_MASKS = "pruning_masks"  # Add this to prevent pruning variables from being stored with the model

x = tf.placeholder("float", shape=[None, 28 * 28])
y_ = tf.placeholder("float", shape=[None, 10])

# fc1
W_fc1 = weight_variable([28 * 28, 300])
prune_mask1 = tf.Variable(tf.ones_like(W_fc1), trainable=False, collections=[tf.GraphKeys.PRUNING_MASKS])
fc1_pruned = tf.mul(W_fc1, prune_mask1)
b_fc1 = bias_variable([300])
h_fc1 = tf.nn.relu(tf.matmul(x, fc1_pruned) + b_fc1)
# Dropout
keep_prob1 = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob1)
# fc2
W_fc2 = weight_variable([300, 100])
prune_mask2 = tf.Variable(tf.ones_like(W_fc2), trainable=False, collections=[tf.GraphKeys.PRUNING_MASKS])
fc2_pruned = tf.mul(W_fc2, prune_mask2)
b_fc2 = bias_variable([100])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, fc2_pruned) + b_fc2)
# Dropout
keep_prob2 = tf.placeholder("float")
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob2)
# fc3
W_fc3 = weight_variable([100, 10])
prune_mask3 = tf.Variable(tf.ones_like(W_fc3), trainable=False, collections=[tf.GraphKeys.PRUNING_MASKS])
fc3_pruned = tf.mul(W_fc3, prune_mask3)
b_fc3 = bias_variable([10])
logits = tf.matmul(h_fc2_drop, fc3_pruned) + b_fc3
y = tf.nn.softmax(logits)

# Define loss function, optimization technique, and accuracy metric
# Add epsilon to prevent 0 log 0; See http://quabr.com/33712178/tensorflow-nan-bug
# http://stackoverflow.com/questions/34240703/difference-between-tensorflow-tf-nn-softmax-and-tf-nn-softmax-cross-entropy-with
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_)
# "Overall, L2 regularization gives the best pruning results."
l2_loss = tf.nn.l2_loss(tf.concat(0, [tf.reshape(W_fc1, [-1]), tf.reshape(W_fc2, [-1]), tf.reshape(W_fc3, [-1])]))
l2_weight_decay = 0.0001  # 0.001 Suggested by Hinton et al. in 2012 ImageNet paper, but smaller works here
loss = cross_entropy + l2_loss * l2_weight_decay
# "After pruning, the network is retrained with 1/10 of the original network's original learning rate."
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Define pruning ops
# Placeholder later; want to set this to a small value greater than zero, else indicator matrix revives pruned neurons!
# "The pruning threshold is chosen as a quality parameter multiplied by the standard deviation of a layer's weights."
threshold = 0.0001
t1 = tf.sqrt(tf.nn.l2_loss(W_fc1)) * threshold
t2 = tf.sqrt(tf.nn.l2_loss(W_fc2)) * threshold
t3 = tf.sqrt(tf.nn.l2_loss(W_fc3)) * threshold
# Apply the previous prune masks each time
indicator_matrix1 = tf.mul(tf.to_float(tf.greater_equal(W_fc1, tf.ones_like(W_fc1) * t1)), prune_mask1)
indicator_matrix2 = tf.mul(tf.to_float(tf.greater_equal(W_fc2, tf.ones_like(W_fc2) * t2)), prune_mask2)
indicator_matrix3 = tf.mul(tf.to_float(tf.greater_equal(W_fc3, tf.ones_like(W_fc3) * t3)), prune_mask3)

# Update the prune masks
update_mask1 = tf.assign(prune_mask1, indicator_matrix1)
update_mask2 = tf.assign(prune_mask2, indicator_matrix2)
update_mask3 = tf.assign(prune_mask3, indicator_matrix3)
update_all_masks = tf.group(update_mask1, update_mask2, update_mask3)

# Applying the pruning mask to the actual weights that are saved
prune_fc1 = W_fc1.assign(fc1_pruned)
prune_fc2 = W_fc2.assign(fc2_pruned)
prune_fc3 = W_fc3.assign(fc3_pruned)
prune_all = tf.group(prune_fc1, prune_fc2, prune_fc3)

# Helper ops
nonzero_indicator1 = tf.to_float(tf.not_equal(W_fc1, tf.zeros_like(W_fc1)))
nonzero_indicator2 = tf.to_float(tf.not_equal(W_fc2, tf.zeros_like(W_fc2)))
nonzero_indicator3 = tf.to_float(tf.not_equal(W_fc3, tf.zeros_like(W_fc3)))
count_parameters1 = tf.reduce_sum(nonzero_indicator1)
count_parameters2 = tf.reduce_sum(nonzero_indicator2)
count_parameters3 = tf.reduce_sum(nonzero_indicator3)

# Create a saver for writing training checkpoints.
saver = tf.train.Saver()

# Run training in a session
sess = tf.Session()
sess.run(tf.initialize_all_variables())
sess.run(tf.initialize_variables(tf.get_collection(tf.GraphKeys.PRUNING_MASKS)))


def print_mask_parameter_counts():
    print("# Mask Parameter Counts")
    print("  - Mask1: {0}".format(
        sess.run(tf.reduce_sum(tf.to_float(tf.not_equal(indicator_matrix1, tf.zeros_like(indicator_matrix1)))))))
    print("  - Mask2: {0}".format(
        sess.run(tf.reduce_sum(tf.to_float(tf.not_equal(indicator_matrix2, tf.zeros_like(indicator_matrix2)))))))
    print("  - Mask3: {0}".format(
        sess.run(tf.reduce_sum(tf.to_float(tf.not_equal(indicator_matrix3, tf.zeros_like(indicator_matrix3)))))))


def print_accuracy():
    feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob1: TEST_KEEP_PROB, keep_prob2: TEST_KEEP_PROB}
    print("test accuracy %g" % sess.run(accuracy, feed_dict=feed_dict))


# "So when we retrain the pruned layers, we should keep the surviving parameters instead of re-initializing them."
# "To prevent this, we fix the parameters for CONV layers and only retrain the FC layers after pruning the FC layers,
#  and vice versa."
def train(iterations=100000, kp1=0.5, kp2=0.5):
    for i in range(iterations):
        batch_xs, batch_ys = mnist.train.next_batch(50)
        if i % 100 == 0:
            feed_dict = {x: batch_xs, y_: batch_ys, keep_prob1: TEST_KEEP_PROB, keep_prob2: TEST_KEEP_PROB}
            train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
            print("step %d training accuracy %g" % (i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob1: kp1, keep_prob2: kp2})
    print_accuracy()


def print_parameter_counts():
    print("# W Parameter Counts")
    print("  - Parameters1: {0}".format(sess.run(count_parameters1)))
    print("  - Parameters2: {0}".format(sess.run(count_parameters2)))
    print("  - Parameters3: {0}".format(sess.run(count_parameters3)))


# "During retraining, however, the dropout ratio must be adjusted to account for the change in model capacity."
def calculate_new_keep_prob(original_keep_prob, original_connections, retraining_connections):
    return 1.0 - ((1.0 - original_keep_prob) * math.sqrt(retraining_connections / original_connections))


def compress(times=1):
    kp1 = kp2 = 0.5
    for i in range(times):
        print("# Compressing iteration {0}...".format(i + 1))
        c1 = sess.run(count_parameters1)
        c2 = sess.run(count_parameters2)
        print_mask_parameter_counts()
        print("# Before pruning")
        print_parameter_counts()
        sess.run(update_all_masks)
        print_mask_parameter_counts()
        sess.run(prune_all)
        c1_retrain = sess.run(count_parameters1)
        c2_retrain = sess.run(count_parameters2)
        kp1 = calculate_new_keep_prob(kp1, c1, c1_retrain)
        kp2 = calculate_new_keep_prob(kp2, c2, c2_retrain)
        # Retrain with pruned connections
        print("# Before retraining")
        print_parameter_counts()
        train(100000, kp1, kp2)
        print("# After retraining")
        print_parameter_counts()
    saver.save(sess, "compressed_model/compressed_model")
    print("# Saved compressed model")
    print_parameter_counts()
    print_accuracy()


if __name__ == '__main__':
    sess.run(tf.initialize_all_variables())
    train()
    saver.save(sess, "uncompressed_model/uncompressed_model")
    compress(5)  # "We used five iterations of pruning an retraining."
    # After 5 iterations of pruning and retraining
    # Test Accuracy:
    #   98.19% => 97.23%
    # Parameters:
    #   fc1: 235,200 => 22,848
    #   fc2: 30,000 => 4,071
    #   fc3: 1,000 => 412
    #   total: ~267k => ~27k (~10x compression)
    # Protobuf Size (after gzip):
    #   ~2.8 MB => ~490kB (~6x compression)
