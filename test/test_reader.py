import unittest
import numpy as np

from core.reader import Reader
from test.values import FromImg
from test.values import FromText
from test.values import Constants
from test.utils import TestCaseWithArrayEqual


class ReaderSetup(TestCaseWithArrayEqual):
    def setUp(self) -> None:
        self.reader = Reader(Constants.small_img)
        
        
class JoinBits(ReaderSetup):
    def test_equality(self) -> None:
        correct_output = FromText.to_ints
        output = self.reader._join_bits(FromText.to_column_bits)
        self.assertArrayEqual(correct_output, output)

    def test_type(self) -> None:
        output = self.reader._join_bits(FromText.to_column_bits)
        self.assertIsInstance(output, np.ndarray)

    def test_dtype(self) -> None:
        correct_output = self.reader.dtype
        output = self.reader._join_bits(FromText.to_column_bits).dtype
        self.assertEqual(correct_output, output)


class FilterLastBits(ReaderSetup):
    def test_equality(self) -> None:
        reader = Reader(FromImg.img, num_encoding_bits=FromImg.num_encoding_bits, num_bits=FromImg.num_bits)
        correct_output = FromImg.masked_first_bits
        output = reader._filter_last_bits(FromImg.img)
        self.assertArrayEqual(correct_output, output)

    def test_type(self) -> None:
        output = self.reader._filter_last_bits(FromImg.img)
        self.assertIsInstance(output, np.ndarray)

    def test_dtype(self) -> None:
        correct_output = np.integer
        output = self.reader._filter_last_bits(FromImg.img).dtype
        self.assertEqual(correct_output, output)


if __name__ == '__main__':
    unittest.main()
