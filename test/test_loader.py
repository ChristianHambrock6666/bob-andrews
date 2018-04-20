from unittest import TestCase
from core.config import Config
import core.loader as ld
import numpy as np

class TestLoader(TestCase):

    def test_event(self):
        event1 = ld.Event(1, 2)
        event2 = ld.Event(1, 3)

        self.assertTrue(event1 == event1, msg='test_event equal failed')
        self.assertFalse(event1 == event2, msg='test_event unequal failed')

    def test_batching(self):
        cf = Config()

        event1 = ld.Event(1, 5)
        event2 = ld.Event(2, 6)
        event3 = ld.Event(3, 7)
        event4 = ld.Event(4, 8)

        test_loader = ld.Loader(cf)

        test_loader.train_events = np.array([event1, event2, event3, event4])

        feature_batch, label_batch = test_loader.get_next_train_batch(2)

        np.testing.assert_array_equal(np.array([1, 2]), feature_batch, err_msg="retrieved features wrong")
        np.testing.assert_array_equal(np.array([5, 6]), label_batch, err_msg="retrieved labels wrong")

        np.testing.assert_array_equal(
            np.array([event3, event4, event1, event2]),
            test_loader.train_events,
            err_msg="remaining events wrong")

    def test_counters(self):
        cf = Config()

        event1 = ld.Event(1, 5)
        event2 = ld.Event(2, 6)
        event3 = ld.Event(3, 7)
        event4 = ld.Event(4, 8)

        loader = ld.Loader(cf)

        loader.train_events = np.array([event1, event2, event3, event4])
        loader.get_next_train_batch(2)
        loader.get_next_train_batch(2)
        loader.get_next_train_batch(2)

        self.assertTrue(loader.epochs == 1, msg='epochs counter failed')
        self.assertTrue(loader.batches == 3, msg='batches counter failed')
        self.assertTrue(loader.events == 6, msg='events counter failed')


    def test__prepare_text_input(self):
        cf = Config()
        cf.string_length = 3

        test_loader = ld.Loader(cf)
        returned_events = test_loader._prepare_text_input("""ficke de""")

        returned_features = [e.feature for e in returned_events]
        returned_labels = [e.label for e in returned_events]

        self.assertTrue(list(np.array(returned_features).shape) == [2, 3, 32], msg='feature shape correctness')
        self.assertTrue(list(np.array(returned_labels).shape) == [2, 2], msg='label shape correctness')





class TestCharTrf(TestCase):
    def test___init__(self):
        test_chartrf = ld.CharTrf(allowed_chars="abc")

        self.assertTrue(test_chartrf.char2num["a"] == 1, msg='translate a')
        self.assertTrue(test_chartrf.num2char[2] == "b", msg='translate 2')
        self.assertTrue(list(test_chartrf.char2one_hot["c"]) == [0.0, 0.0, 0.0, 1.0], msg='translate c')

    def test_string_to_tensor(self):
        test_chartrf = ld.CharTrf(allowed_chars="abc")
        test_tensor = test_chartrf.string_to_tensor("ac d")
        expected_tensor = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

        self.assertTrue(np.array_equal(test_tensor, expected_tensor), msg='translate string to tensor')

    def test_tensor_to_string(self):
        test_chartrf = ld.CharTrf(allowed_chars="abc", default_char='$')
        test_tensor = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        test_string = test_chartrf.tensor_to_string(test_tensor)

        self.assertTrue(test_string == "ac$$", msg='translate tennsor to string')