import lime
import lime.lime_tabular
import tensorflow as tf
import numpy as np


class Evaluator(object):
    """wip class to interpret the output of the """

    def __init__(self, config, network, char_trf):
        self.ct = char_trf
        self.cf = config
        self.network = network

        self.x = None
        self.initialize_graph()
        self.cf.allowed_chars = "abcdefghijklmnopqrstuvwxyzäöüß_"
        self.cf.string_length = 200
        self.categorical_features = range(self.cf.string_length)

        self.feature_names = list(self.cf.allowed_chars).append("-")
        self.categorical_names = {}
        for len_sentence in range(self.cf.n_chars):
            self.categorical_names[len_sentence] = self.feature_names

    def initialize_graph(self):
        """build graph"""
        self.x = tf.placeholder(tf.float32, [None, self.cf.string_length, self.cf.n_chars])
        self.pred = self.network.predict(self.x)

    def predict(self, sess, tensor):
        return sess.run(self.pred, feed_dict={self.x: [tensor]})[0]

    def importanize_tensor_sentence(self, sess, tensor):
        """check character importance for sentence. tensor_batch: [batch_size, num_letters, num_chars]"""
        pred0 = sess.run(self.pred, feed_dict={self.x: [tensor]})[0]  # the [0] because of batch processing
        character_importance = []

        for letter_idx in range(len(tensor)):
            removed_tensor = tensor.copy()
            removed_tensor[letter_idx] = tensor[letter_idx] * 0.0
            removed_tensor[letter_idx][0] = 1.0
            pred_removed = sess.run(self.pred, feed_dict={self.x: [removed_tensor]})[0]

            diff_to_pred0 = pred0[1] * (max(pred0[1] - pred_removed[1], 0) + max(pred_removed[0] - pred0[0],
                                                                                 0))  # something like the probability change after taking out the respective character
            character_importance.append(diff_to_pred0)

        return character_importance, pred0

    def setup_lime_explainer(self, sess, train_data):
        data = train_data

        for j, names in enumerate(self.categorical_names):
            data[:, j] = np.array(list(map(lambda x: x, data[:, j])))

        data = data.astype(float)

        return None

    def importanize_tensor_sentence_lime(self, sess, train_data):
        """check character importance for sentence. tensor_batch: [batch_size, num_letters, num_chars]"""

        data = np.array(list(map(lambda x: x, sentence_list)))
        data = data.astype(float)

        pred_fn = lambda x: sess.run(self.pred, feed_dict={self.x: [self.ct.string_to_tensor(''.join(sentence_list))]})[
            0].astype(float)

        explainer = lime.lime_tabular.LimeTabularExplainer(train, class_names=['false', 'true'],
                                                           feature_names=self.feature_names,
                                                           categorical_features=self.categorical_features,
                                                           categorical_names=self.categorical_names, kernel_width=3,
                                                           verbose=False)

        i = 127
        exp = explainer.explain_instance(test[i], pred_fn, num_features=5)
        exp.save_to_file("test")
