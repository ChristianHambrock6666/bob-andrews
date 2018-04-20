import tensorflow as tf


class Network:
    """'dims: [batch, height, width, channels]"""
    weights = None
    biases = None

    def __init__(self, config):
        self.cf = config
        self.len_layer1_out = self._how_many_strides_fit(self.cf.string_length, self.cf.syllable_length, self.cf.strides1)
        self.len_layer2_out = self._how_many_strides_fit(self.len_layer1_out, self.cf.word_length, self.cf.strides2)
        self._initialize_params_()

    def _how_many_strides_fit(self, tot_length, window_size, stride_width):
        """calculate how many non-padded strides fit"""
        return int((tot_length - window_size) / stride_width) + 1

    def _initialize_params_(self):
        """initialize all parameters which are later optimized by tensorflow"""
        rand_std = 0.01

        self.weights = {
                 'w_conv1': tf.Variable(tf.random_normal(
                     [self.cf.syllable_length, self.cf.n_chars, 1, self.cf.n_syllables], stddev=rand_std)),
                 'w_conv2': tf.Variable(tf.random_normal(
                     [self.cf.word_length, 1, self.cf.n_syllables, self.cf.n_words], stddev=rand_std)),
                 'w_fc': tf.Variable(tf.random_normal(
                     [self.len_layer2_out * self.cf.n_words, self.cf.output_number], stddev=rand_std)),
                 'out': tf.Variable(tf.random_normal(
                     [self.cf.output_number, self.cf.n_classes], stddev=rand_std))
             }

        self.biases = {
                 'b_fc': tf.Variable(tf.random_normal([self.cf.output_number])),
                 'out': tf.Variable(tf.random_normal([self.cf.n_classes]))
             }

    def predict(self, input_features):
        """glue together all layers to transform features to predictions"""
        out_layer0 = self.layer0(input_features)
        out_layer1 = self.layer1(out_layer0)
        out_layer2 = self.layer2(out_layer1)
        out_layer3 = self.layer3(out_layer2)
        out_layer4 = self.layer4(out_layer3)

        return out_layer4

    def layer0(self, in_tensor):
        """Reshape to make standard input (picture recognition), last is the possible color channel"""
        return tf.reshape(in_tensor, shape=[-1, self.cf.string_length, self.cf.n_chars, 1])  # [batch, chars, char_classes, 'channels']

    def layer1(self, in_tensor):
        """convolutional layer to recognize 'syllables'"""
        return tf.nn.relu(tf.nn.conv2d(
            in_tensor,
            self.weights['w_conv1'],
            strides=[1, self.cf.strides1, self.cf.n_chars, 1],  # [batch, chars_in_string, char_classes, 'channels']
            padding="VALID"))

    def layer2(self, in_tensor):
        """convolutional layer to recognize 'words' made up of 'syllables'"""
        return tf.nn.relu(tf.nn.conv2d(
            in_tensor, self.weights['w_conv2'],
            strides=[1, self.cf.strides2, self.cf.n_syllables, 1],  # [batch, cnn1_output, syllables, 'channels']
            padding="VALID"))

    def layer3(self, in_tensor):
        """flatten everything and make a standard feed forward layer"""
        fc_input = tf.reshape(in_tensor, [-1, self.len_layer2_out * self.cf.n_words])  # [batch, giant vector]
        return tf.matmul(fc_input, self.weights['w_fc']) + self.biases['b_fc']

    def layer4(self, in_tensor):
        """output feed forward layer"""
        return tf.nn.sigmoid(tf.matmul(in_tensor, self.weights['out']) + self.biases['out'])

