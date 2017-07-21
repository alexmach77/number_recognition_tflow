import tensorflow as tf
import cv2
#image = misc.imread('/Users/Alex/Google Drive/Workshops/europython/flask_ml/tutorial/webapp/static_files/export.png')


# Network parameters
n_classes = 10  # MNIST total classes (0-9 digits)

weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32]), name='Wc1'),
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64]), name='Wc2'),
    'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024]), name='Wfc'),
    'out': tf.Variable(tf.random_normal([1024, n_classes]), name='Wo')
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32]), name='bc1'),
    'bc2': tf.Variable(tf.random_normal([64]), name='bc2'),
    'bd1': tf.Variable(tf.random_normal([1024]), name='bfc'),
    'out': tf.Variable(tf.random_normal([n_classes]), name='bo')
}



def conv2d(x, W, b, strides=1, name='conv'):
    with tf.name_scope(name):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        x = tf.nn.relu(x)
    return x


def maxpool2d(x, k=2, name='max_pool'):
    with tf.name_scope(name):
        x = tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    return x


def conv_net(x, weights, biases, dropout):
    # Layer 1 - 28x28x1 to 14x14x32
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], name='conv1')
    conv1 = maxpool2d(conv1, k=2, name='max_pool1')

    # Layer 2 - 14x14x32 to 7x7x64
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], name='conv2')
    conv2 = maxpool2d(conv2, k=2, name='max_pool2')

    # Fully conected Layer - 7x7x64 to 1024
    with tf.name_scope('fc1'):
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, dropout)

    with tf.name_scope('fc2'):
        # Output Layer - class prediction - 1024 to 10
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out


# tf Graph input
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
keep_prob = tf.placeholder(tf.float32)  # keep probability (dropout)
# Model
logits = conv_net(x, weights, biases, keep_prob)
probs = tf.nn.softmax(logits)
saver = tf.train.Saver()

def evaluate():
    image = cv2.imread('/Users/Alex/Google Drive/Workshops/europython/flask_ml/tutorial/webapp/static_files/export.png')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image / 255
    image = image.reshape([1, 28, 28, 1])

    print(type(image))
    print(image.shape)

    with tf.Session() as sess:
        #saver = tf.train.import_meta_graph("/Users/Alex/Google Drive/Workshops/europython/flask_ml/tutorial/webapp/model.ckpt.meta")
        saver.restore(sess, "/Users/Alex/Google Drive/Workshops/europython/flask_ml/tutorial/webapp/model.ckpt")

        probs_value = sess.run(probs, feed_dict={
            x: image,
            keep_prob: 1.})

        #prediction = tf.argmax(probs)
        print(probs_value)
        print(probs_value.argmax())
        float_formatter = lambda x: "%.2f" % x
        print(float_formatter(probs_value[0][7]))
        return probs_value.argmax()