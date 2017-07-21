import tensorflow as tf
import cv2

tf.reset_default_graph() 

saver = tf.train.import_meta_graph('./models/model_98_43.ckpt.meta')

inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')
keep_prob = tf.get_default_graph().get_tensor_by_name('keep_prob:0')
#print('The input placeholder is expecting an array of shape {} and type {}'.format(inputs.shape, inputs.dtype))

logits = tf.get_default_graph().get_tensor_by_name('logits:0')
probs = tf.nn.softmax(logits)


def evaluate():
    image = cv2.imread('./static_files/export.png')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image / 255 # normalize pixel values to [0,1]
    image = 1 - image # invert colors -> black background
    image = image.reshape([1, 28, 28, 1])
    # print(image.shape)

    with tf.Session() as sess:
        saver.restore(sess, "./models/model_98_43.ckpt")

        probs_value = sess.run(probs, feed_dict={
            inputs: image,
            keep_prob: 1.})

        print(probs_value)
        prediction = tf.argmax(probs)

        return prediction