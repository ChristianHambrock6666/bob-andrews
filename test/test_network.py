from unittest import TestCase

import numpy as np
import tensorflow as tf

from core.network import Network
from core.config import Config


class TestNetwork(TestCase):

    def test_layer_sizes(self):
        cf = Config()
        cf.n_chars = 33
        cf.string_length = 199
        cf.n_syllables = 11
        cf.syllable_length = 3
        cf.word_length = 2
        cf.n_words = 7
        cf.output_number = 101
        cf.n_classes = 11
        cf.strides1 = 1
        cf.strides2 = 1

        test_netowrk = Network(cf)

        # layer 0 --------------------------
        input_to_layer = tf.placeholder(tf.float32, [None, 199, 33])
        out_layer = test_netowrk.layer0(input_to_layer)
        out_layer_shape = out_layer.get_shape().as_list()
        self.assertTrue(out_layer_shape == [None, 199, 33, 1], msg="layer 0 shape comp")


        # layer 1 --------------------------
        input_to_layer = tf.placeholder(tf.float32, [None, 199, 33, 1])
        out_layer = test_netowrk.layer1(input_to_layer)
        out_layer_shape = out_layer.get_shape().as_list()
        self.assertTrue(out_layer_shape == [None, 197, 1, 11], msg="layer 1 shape comp")

        # layer 2 --------------------------
        input_to_layer = tf.placeholder(tf.float32, [None, 197, 1, 11])
        out_layer = test_netowrk.layer2(input_to_layer)
        out_layer_shape = out_layer.get_shape().as_list()
        self.assertTrue(out_layer_shape == [None, 196, 1, 7], msg="layer 2 shape comp")

        # layer 3 --------------------------
        input_to_layer = tf.placeholder(tf.float32, [None, 196, 1, 7])
        out_layer = test_netowrk.layer3(input_to_layer)
        out_layer_shape = out_layer.get_shape().as_list()
        self.assertTrue(out_layer_shape == [None, 101], msg="layer 3 shape comp")

        # layer 4 --------------------------
        input_to_layer = tf.placeholder(tf.float32, [None, 101])
        out_layer = test_netowrk.layer4(input_to_layer)
        out_layer_shape = out_layer.get_shape().as_list()
        self.assertTrue(out_layer_shape == [None, 11], msg="layer 3 shape comp")



#sess = tf.InteractiveSession()
#
#a=tf.reshape(tf.constant([1,2,3,4]), [2,2,1])
#b=a.eval()
