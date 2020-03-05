import unittest
import numpy as np

from core.reader import Reader
from core.writer import Writer
from core.converter import Converter
from test.values import Constants
from test.utils import TestCaseWithArrayEqual


class ReaderSetup(TestCaseWithArrayEqual):
    def setUp(self) -> None:
        self.reader = Reader(Constants.small_img)


class ReadHiddenBytesTests(ReaderSetup):
    def test_equality(self) -> None:
        txt = Constants.char_pool
        converter = Converter()
        encoded_txt = converter.encode(txt)
        encoded_img = Writer(Constants.big_img).insert_bytes(encoded_txt)
        encoded_rebuilt_txt = Reader(encoded_img).read_hidden_bytes()
        decoded_rebuilt_txt = converter.decode(encoded_rebuilt_txt)
        self.assertIn(txt, decoded_rebuilt_txt)

    def test


if __name__ == '__main__':
    unittest.main()
