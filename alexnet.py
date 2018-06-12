import tensorflow as tf
import numpy as np
import util

class AlexNet(object):
    def __init__(self, x, keep_prob, num_classes, batch_size, image_size, channels):
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.BATCH_SIZE = batch_size
        self.IMAGE_SIZE = image_size
        self.CHANEELS = channels

        self.create()

    def create(self):
        self.X = tf.reshape(self.X, shape=[-1, self.IMAGE_SIZE, self.IMAGE_SIZE, self.CHANEELS])
        conv1 = util.conv_layer(self.X, ksize=[11, 11, self.CHANEELS, 96], strides=[1, 4, 4, 1], name='conv1')
        pool1 = util.max_pool_layer(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], name='pool1')

        conv2 = util.conv_layer(pool1, ksize=[5, 5, 96, 256], strides=[1, 1, 1, 1], name='conv2', b_value=1., padding='SAME', group=2)
        pool2 = util.max_pool_layer(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], name='pool2')

        conv3 = util.conv_layer(pool2, ksize=[3, 3, 256, 384], strides=[1, 1, 1, 1], name='conv3', padding='SAME')

        conv4 = util.conv_layer(conv3, ksize=[3, 3, 384, 384], strides=[1, 1, 1, 1], name='conv4', b_value=1., padding='SAME', group=2)

        conv5 = util.conv_layer(conv4, ksize=[3, 3, 384, 256], strides=[1, 1, 1, 1], name='conv5', b_value=1., padding='SAME', group=2)
        pool5 = util.max_pool_layer(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], name='pool5')

        fc6 = util.full_connected_layer(pool5, 4096, name='fc6', b_value=1.)
        fc6_dropout = util.dropout(fc6, self.KEEP_PROB)

        fc7 = util.full_connected_layer(fc6_dropout, 4096, name='fc7', b_value=1.)
        fc7_dropout = util.dropout(fc7, self.KEEP_PROB)

        self.fc8 = util.full_connected_layer(fc7_dropout, self.NUM_CLASSES, name='fc8', relu=False)

