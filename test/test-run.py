from unittest import TestCase

import numpy as np
import tensorflow as tf

from core.run import Network


#x = tf.placeholder(tf.float32, [None, n_chars, len_string])
#y = tf.placeholder(tf.float32, [None, 1])


class TestNetwork(TestCase):

    def test__build_feed_dict_(self):
        features = [
            [[1., 2., 3.], [4., 5., 6.]],
            [[7., 8., 9.], [10., 11., 12.]],
            [[13., 14., 15.], [16., 17., 18.]],
            [[19., 20., 21.], [22., 23., 24.]]
        ]
        labels = [[1], [2], [3], [4]]

        network = Network()

        network._initialize_placeholders_(n_chars=2, len_string=3)

        with tf.Session() as sess:
            [features_from_network, labels_from_network] = sess.run([network.placeholders['features'], network.placeholders['labels']], feed_dict= network.build_feed_dict(features=features, labels=labels) )


        np.testing.assert_array_equal(features, features_from_network)
        np.testing.assert_array_equal(labels, labels_from_network)