import tensorflow as tf
from core import loader


class Config:
    """Contains static configurations"""

    def __init__(self):

        self.n_syllables = 30  # number of patterns in first layer, which is a combination of some characters, i.e., something like a syllable
        self.syllable_length = 3    # number of characters in 'syllable'

        self.n_words = 20      # number of patterns which are combined 'syllables'
        self.word_length = 2  # number of 'syllables' in each pattern

        self.output_number = 100  # dimension of fully connected pre-output layer

        self.n_classes = 2  # final classes to predict

        self.epochs = 50  # number of epochs (epoch = whole train data processed) to train
        self.shuffle = True  # True if after each epoch train data is shuffled

        self.batch_size = 50

        self.learning_rate = 0.001

        self.strides1 = 3  # strides in the first layer (chars to 'syllables')
        self.strides2 = 2  # strides in the second layer ('syllables' to 'words')

        self.allowed_chars = "abcdefghijklmnopqrstuvwxyzäöüß "
        self.default_char = '-'  # set this if default char is translated back from tensor
        self.string_length = 200
        self.n_chars = len(self.allowed_chars) + 1  # +1 being the default class

        self.url_train_data = '../data/train_data.txt'
        self.url_test_data = '../data/test_data.txt'

        self.info_patterns = ["nicht", "schloss", "erfahren"]

        self.sigma_chars = 150  # if a random string is drawn it has length string length - normal distribution with std sigma chars

    def to_tex(self):
        """just latex code with the config parameters explained"""

        tex_string = """
        General parameters of the config:
        \n
        \\begin{itemize}
        \\item[{\\bf epochs:}] """ + str(self.epochs) + """
        \\item[{\\bf batch size:}] """ + str(self.batch_size) + """
        \\item[{\\bf shuffle:}] """ + str(self.shuffle) + """
        \\item[{\\bf learning rate:}] """ + str(self.learning_rate) + """
        \\end{itemize}
        \n
        """

        tex_string = tex_string + """
        Data description parameters of the config:
        \n
        \\begin{itemize}
        \\item[{\\bf allowed chars:}] """ + self.allowed_chars + """
        \\item[{\\bf number of targets:}] """ + str(self.n_classes) + """
        \\item[{\\bf number of character classes:}] """ + str(self.n_chars) + """ (one more than char count for the generic class)
        \\end{itemize}
        \n
        """

        tex_string = tex_string + """
        Network description parameters of the config:
        \n
        \\begin{itemize}
        \\item[{\\bf n syllables:}] """ + str(self.n_syllables) + """ number of patterns in first layer, which is a combination of some characters, i.e., something like a syllable
        \\item[{\\bf syllable length:}] """ + str(self.syllable_length) + """ number of characters in 'syllable'
        \\item[{\\bf n words:}] """ + str(self.n_words) + """ number of 'word' patterns which are combined 'syllables'
        \\item[{\\bf word length:}] """ + str(self.word_length) + """ number of 'syllables' in each 'word' pattern
        \\item[{\\bf output number:}] """ + str(self.word_length) + """ dimension of fully connected pre-output layer
        \\item[{\\bf strides 1:}] """ + str(self.strides1) + """ strides in the first layer along the 'sentence'
        \\item[{\\bf strides 2:}] """ + str(self.strides2) + """ strides in the second layer along the 'syllables'
        \\end{itemize}
        \n
        """

        return tex_string