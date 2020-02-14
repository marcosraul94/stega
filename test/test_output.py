import unittest
import numpy as np

from core.reader import Reader
from core.writer import Writer
from core.config import DEFAULT_CONFIG


class Output:
    text = 'aá&^'
    to_bytearray = b'a\xc3\xa1&^'
    to_ints = np.array([97, 195, 161, 38, 94])
    to_binary = np.array(['0b1100001', '0b11000011', '0b10100001', '0b100110', '0b1011110'])
    to_uniform_bytes = np.array(['01100001', '11000011', '10100001', '00100110', '01011110'])


class OutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.output = Output()
        self.reader = Reader(np.arange(1))
        self.writer = Writer(np.arange(1))
        self.config = DEFAULT_CONFIG

    def test_bytearray(self) -> None:
        self.assertEqual(self.output.to_bytearray, bytearray(self.output.text, encoding=self.config.encoding))

    def test_ints(self) -> None:
        equal = np.array_equal(self.output.to_ints, self.writer._split_into_ints(self.output.to_bytearray))
        self.assertTrue(equal)

    def test_binary(self) -> None:
        equal = np.array_equal(self.output.to_binary, self.writer._to_binary(self.output.to_ints))
        self.assertTrue(equal)

    def test_uniform_bytes(self) -> None:
        equal = np.array_equal(self.output.to_uniform_bytes, self.writer._uniform_bytes(self.output.to_binary))