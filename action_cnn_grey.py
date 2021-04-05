from Dataset import KTH
import tensorflow as tf
import numpy as np
import json
import datetime

STEPS = 12000
BATCH_SIZE = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0005
KEEP_PROB = 0.9

def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

def max_pool_2x2x2(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[4]])
    return tf.nn.relu(conv3d(input, W) + b)

def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b

kth = KTH()
kth.load_from_file()
kth.normalize()
# kth.load_specific(list(range(1, 21)), list(range(21, 26)))
# kth.drop_one_case()
history = []


x = tf.compat.v1.placeholder(tf.float32, shape=[None, 60, 80, 8, 1])
y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 6])
keep_prob = tf.compat.v1.placeholder(tf.float32)

learning_rate = tf.Variable(LEARNING_RATE, dtype=tf.float32)
update_learning_rate = tf.compat.v1.assign(learning_rate, learning_rate * (1 - WEIGHT_DECAY))

conv1 = conv_layer(x, shape=[7, 7, 3, 1, 64])
conv1_pool = max_pool_2x2x2(conv1)
conv1_pool_drop = tf.nn.dropout(conv1_pool, rate = 1 - keep_prob)

conv2 = conv_layer(conv1_pool_drop, shape=[7, 7, 3, 64, 128])
conv2_pool = max_pool_2x2x2(conv2)
conv2_flat = tf.reshape(conv2_pool, [-1, 15 * 20 * 2 * 128])

full_1 = tf.nn.relu(full_layer(conv2_flat, 256))
full_1_drop = tf.nn.dropout(full_1, rate = 1 - keep_prob)

full_2 = tf.nn.relu(full_layer(full_1_drop, 128))
full_2_drop = tf.nn.dropout(full_2, rate = 1 - keep_prob)

y_conv = full_layer(full_2_drop, 6)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = y_conv, labels = y_))

train_step = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def test(sess):
    # test_baches = 10
    # X = kth.test_images[:,:,:,:,:1].reshape(test_baches, 100, 60, 80, 8, 1)
    # Y = kth.test_labels.reshape(test_baches, 100, 6)
    # acc = np.mean([sess.run(accuracy, feed_dict={x: X[i], y_: Y[i], keep_prob: 1.0}) for i in range(test_baches)])
    # print("Accuracy: {:.5}%".format(acc * 100))

    random_seed = np.arange(kth.test_images.shape[0])
    np.random.shuffle(random_seed)

    test_images = kth.test_images[random_seed][:1000]
    test_labels = kth.test_labels[random_seed][:1000]

    test_baches = int(len(test_images) / 100)
    X = test_images[:,:,:,:,:1].reshape(test_baches, 100, 60, 80, 8, 1)
    Y = test_labels.reshape(test_baches, 100, 6)
    acc = np.mean([sess.run(accuracy, feed_dict={x: X[i], y_: Y[i], keep_prob: 1.0}) for i in range(test_baches)])
    print("Accuracy: {:.5}%".format(acc * 100))

    return acc * 100

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(STEPS):
        batch = kth.next_batch_grey(BATCH_SIZE)
        updated_LR = sess.run(update_learning_rate)
        if i % 100 == 0:
            print(i,': ', sep='', end='')
            acc = test(sess)
            if i == 0:
                print('   Learning Rate:', updated_LR)
            elif i < 1000:
                print('     Learning Rate:', updated_LR)
            else:
                print('      Learning Rate:', updated_LR)

            C_E = sess.run(cross_entropy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})

            if i == 0:
                print('   Cross entropy:', C_E, '\n')
            elif i < 1000:
                print('     Cross entropy:', C_E, '\n')
            else:
                print('      Cross entropy:', C_E, '\n')

            history.append((acc, C_E))

            if i % 1000 == 0 and not i == 0:
                with open('program_runs/' + str(datetime.datetime.now()).replace(':', '-') + '_grey.txt', 'w') as f:
                    f.write('BATCH_SIZE = ' + str(BATCH_SIZE))
                    f.write('\nLEARNING_RATE = ' + str(LEARNING_RATE))
                    f.write('\nWEIGHT_DECAY = ' + str(WEIGHT_DECAY))
                    f.write('\nKEEP_PROB = ' + str(KEEP_PROB))
                    f.write('\n')
                    # print(history)
                    f.write(json.dumps(str(history)))
        # else:
        #     print(i)

        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: KEEP_PROB})

    test(sess)

with open('program_runs/' + str(datetime.datetime.now()).replace(':', '-') + '_grey.txt', 'w') as f:
    f.write('BATCH_SIZE = ' + str(BATCH_SIZE))
    f.write('\nLEARNING_RATE = ' + str(LEARNING_RATE))
    f.write('\nWEIGHT_DECAY = ' + str(WEIGHT_DECAY))
    f.write('\nKEEP_PROB = ' + str(KEEP_PROB))
    f.write('\n')
    # print(history)
    f.write(json.dumps(str(history)))