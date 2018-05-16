# -*- coding: utf-8 -*-
import os
import re
import struct
from random import randint

import numpy as np
from array import array
import tensorflow as tf

from io import BytesIO, StringIO


class Event(object):
    """container for feature target pairs"""

    def __init__(self, feature, label):
        self.feature = feature
        self.label = label

    def __eq__(self, other):
        """Equality yo!"""
        if isinstance(other, self.__class__):
            return (self.feature == other.feature) and (self.label == other.label)
        else:
            return False

    def __ne__(self, other):
        """Some things are more equal than others."""
        return not self.__eq__(other)


class Loader(object):
    """Data handling from input txt file"""

    def __init__(self, config):

        self.cf = config

        self.ct = CharTrf(config)

        with open(self.cf.url_train_data, 'r') as f:
            self.train_text = re.sub(r'\W+', " ", str(f.read()).lower())
            self.train_text = re.sub(r'[^' + self.cf.allowed_chars + ']', "-", self.train_text)
        with open(self.cf.url_test_data, 'r') as f:
            self.test_text = re.sub(r'\W+', " ", str(f.read()).lower())
            self.test_text = re.sub(r'[^' + self.cf.allowed_chars + ']', "-", self.test_text)

        self.train_text_length = len(self.train_text)
        self.train_events = self._prepare_text_input(self.train_text)
        self.test_events = self._prepare_text_input(self.train_text)

        self.epochs = 0
        self.batches = 0
        self.events = 0

        self.new_epoch = True

    def _prepare_text_input(self, text_content):
        """Takes text as a single long String and builds features and labels."""
        events = []

        s = StringIO(text_content)
        chunk = s.read(self.cf.string_length)
        while len(chunk) == self.cf.string_length:
            chunk_vector_rep = self.ct.string_to_tensor(chunk)
            has_search_terms = ("nicht" in chunk)  # ("schloß" in chunk) or ("landvermesser" in chunk) or
            chunk_label = [1 - has_search_terms, has_search_terms]

            events.append(Event(chunk_vector_rep, chunk_label))

            chunk = s.read(self.cf.string_length)
        return np.array(events)

    def get_next_train_batch(self, batch_size=32, shuffle=False):
        """Generates next train batch and starts from beginning after one epoch (shuffle if wanted)"""

        self.new_epoch = False

        train_batch_events = self.train_events[:batch_size]
        self.train_events = np.roll(self.train_events, batch_size)
        features = np.array([e.feature for e in train_batch_events], np.float32)
        labels = np.array([e.label for e in train_batch_events], np.float32)

        self.update_processed_state(batch_size)

        if shuffle and self.new_epoch:
            np.random.shuffle(self.train_events)
        return features, labels

    def get_random_string(self, text):
        substract = max(int(abs(np.random.normal(scale=self.cf.sigma_chars))), 0)
        str_length = self.cf.string_length - substract
        str_start = randint(0, self.train_text_length - str_length)
        return text[str_start:str_start + str_length]

    def update_processed_state(self, batch_size):
        if self.events % len(self.train_events) > (self.events + batch_size) % len(self.train_events):
            self.epochs += 1
            self.new_epoch = True
        self.batches += 1
        self.events += batch_size

    def get_next_train_batch_sample(self, batch_size=32):
        """Generates next train batch by sampling from text random (with varying length)"""
        self.new_epoch = False
        features = np.empty((0, self.cf.string_length, self.ct.n_char_classes), np.float32)
        labels = np.empty((0, self.cf.n_classes), np.float32)
        for i in range(batch_size):
            event = self.ct.string_to_event(self.get_random_string(self.train_text))
            features = np.append(features, [event.feature], axis=0)
            labels = np.append(labels, [event.label], axis=0)

        self.update_processed_state(batch_size)
        return features, labels

    def get_test_data(self):
        """just give out the whole test data in one batch"""

        features = np.array([event.feature for event in self.test_events], np.float32)
        labels = np.array([event.label for event in self.test_events], np.float32)

        return features, labels

    def get_train_sentence_char_lists(self):
        return [list(self.ct.tensor_to_string(event.feature)) for event in self.train_events]


class CharTrf(object):
    """Translate strings to tensors (i.e., np.array) and back"""

    def __init__(self, config):
        self.n_char_classes = len(config.allowed_chars) + 1
        self.default_char = config.default_char
        self.cf = config

        self.allowed_chars = config.allowed_chars

        self.char2num = {char: idx + 1 for idx, char in enumerate(self.allowed_chars)}
        self.num2char = {v: k for k, v in self.char2num.items()}
        self.char2one_hot = {k: self.one_hot(v) for k, v in self.char2num.items()}

    def one_hot(self, idx):
        """generate one hot vector of appropriate size (n_char_classes)"""
        zeros = np.zeros(self.n_char_classes)
        np.put(zeros, idx, 1)
        return np.array(zeros, np.float32)

    def string_to_tensor(self, in_string):
        return np.array([self.char_to_one_hot(char) for char in in_string], np.float32)

    def tensor_to_numbers(self, in_tensor):
        return [float(np.argmax(vec)) for vec in in_tensor]

    def numbers_to_tensor(self, in_numbers):
        return [self.one_hot(num) for num in in_numbers]

    def tensor_to_string(self, in_tensor):
        numbers = [np.argmax(vec) for vec in in_tensor]
        return self.indices_to_string(numbers)

    def indices_to_string(self, in_numbers):
        return ''.join([self.num2char.get(num, self.default_char) for num in in_numbers])

    def char_to_one_hot(self, char):
        return self.char2one_hot.get(char, self.one_hot(0))

    def string_to_event(self, in_string):
        """Takes text as a single String and builds feature and label."""
        has_search_terms = self.contains_pattern(in_string)
        str_out = in_string.ljust(self.cf.string_length)  # padding
        if len(str_out) > self.cf.string_length:
            str_out = str_out[0:self.cf.string_length]
        chunk_label = np.array([1 - has_search_terms, has_search_terms], np.float32)
        return Event(self.string_to_tensor(str_out), chunk_label)

    def contains_pattern(self, in_string):
        has_search_terms = False
        for p in self.cf.info_patterns:
            has_search_terms = (p in in_string)
        return has_search_terms
