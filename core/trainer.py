import tensorflow as tf
import sys
import os
from os import listdir

class Trainer(object):

    def __init__(self, config, network):
        self.cf = config
        self.network = network
        self.batch_cnt = 0

        self.x = None
        self.y = None
        self.train_output = None
        self.test_output = None
        self._initialize_graph()  # STATE!
        self._initialize_tensorboard()  # STATE!


    def _initialize_graph(self):
        """build graph"""
        self.x = tf.placeholder(tf.float32, [None, self.cf.string_length, self.cf.n_chars])
        self.y = tf.placeholder(tf.float32, [None, self.cf.n_classes])

        self.pred = self.network.predict(self.x)
        self.cost = self._loss_(self.pred, self.y)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.cf.learning_rate).minimize(self.cost)  #

        correct = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    def _initialize_tensorboard(self):
        tf.summary.scalar("train_accuracy", self.accuracy)
        tf.summary.scalar("train_cost", self.cost)
        tf.summary.scalar("test_accuracy", self.accuracy)
        tf.summary.scalar("test_cost", self.cost)
        self.train_merged = tf.summary.merge_all()
        self.test_merged = tf.summary.merge_all()

    def _loss_(self, prediction, y):
        """loss function only taken out of initialize_graph because of more singular importance... private, dont touch, dont need to!"""
        #single_loss = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)
        single_loss = tf.pow(prediction - y, 2)
        return tf.reduce_mean(single_loss)

    def train(self, sess, batch_x, batch_y, train_writer):
        """run the graph build in the constructor"""
        self.batch_cnt = self.batch_cnt + 1
        self.train_output = sess.run([self.optimizer, self.cost, self.network.weights, self.accuracy, self.train_merged], feed_dict={self.x: batch_x, self.y: batch_y})
        train_writer.add_summary(self.train_output[4], self.batch_cnt)
        return self.train_output

    def test(self, sess, x, y):
        """same as train, only do not include the optimizer to not move the weights when generating output"""
        self.test_output = sess.run([self.cost, self.accuracy], feed_dict={self.x: x, self.y: y})
        return self.test_output


    def print_info_(self):
        """wip"""
        print("--------------------------------------------------------------")
        print("cost", self.train_output[1])
        print("batch avg accuracy: ", self.train_output[3])

