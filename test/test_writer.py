import unittest
import numpy as np

from core.writer import Writer
from core.reader import Reader
from test.values import Constants
from test.values import FromImg
from test.values import FromText
from core.converter import Converter
from test.utils import TestCaseWithArrayEqual


class WriterSetup(TestCaseWithArrayEqual):
    def setUp(self) -> None:
        self.writer = Writer(Constants.small_img)
        self.converter = Converter()


class InsertBytesTests(WriterSetup):
    def test_equality(self) -> None:
        txt = Constants.char_pool
        converter = Converter()
        encoded_txt = converter.encode(txt)
        encoded_img = Writer(Constants.big_img).insert_bytes(encoded_txt)
        encoded_rebuilt_txt = Reader(encoded_img).read_hidden_bytes()
        decoded_rebuilt_txt = converter.decode(encoded_rebuilt_txt)
        self.assertIn(txt, decoded_rebuilt_txt)

    def test_type(self) -> None:
        output = self.writer.insert_bytes(FromText.to_bytearray)
        self.assertIsInstance(output, np.ndarray)

    def test_dtype(self) -> None:
        correct_output = self.writer.dtype
        output = self.writer.insert_bytes(FromText.to_bytearray).dtype
        self.assertEqual(correct_output, output)


class PrepareImgTests(WriterSetup):
    def test_equality(self) -> None:
        writer = Writer(
            FromImg.img, num_bits=FromImg.num_bits, num_encoding_bits=FromImg.num_encoding_bits
        )
        correct_output = FromImg.masked_last_bits
        output = writer._prepare_img()
        self.assertArrayEqual(correct_output, output)

    def test_type(self) -> None:
        output = self.writer._prepare_img()
        self.assertIsInstance(output, np.ndarray)

    def test_dtype(self) -> None:
        correct_output = self.writer.dtype
        output = self.writer._prepare_img().dtype
        self.assertEqual(correct_output, output)


class SplitIntoIntsTests(WriterSetup):
    def test_equality(self) -> None:
        writer = Writer(Constants.small_img, dtype=Constants.dtype)
        correct_output = FromText.to_ints
        output = writer._split_into_ints(FromText.to_bytearray)
        self.assertArrayEqual(correct_output, output)

    def test_type(self) -> None:
        output = self.writer._split_into_ints(FromText.to_bytearray)
        self.assertIsInstance(output, np.ndarray)

    def test_dtype(self) -> None:
        correct_output = self.writer.dtype
        output = self.writer._split_into_ints(FromText.to_bytearray).dtype
        self.assertEqual(correct_output, output)


class ToBinaryTests(WriterSetup):
    def test_equality(self) -> None:
        correct_output = FromText.to_binary
        output = self.writer._to_binary(FromText.to_ints)
        self.assertArrayEqual(correct_output, output)

    def test_type(self) -> None:
        output = self.writer._to_binary(FromText.to_ints)
        self.assertIsInstance(output, np.ndarray)

    def test_dtype(self) -> None:
        output = self.writer._to_binary(FromText.to_ints).dtype.type
        self.assertIs(output, np.str_)


class UniformBytesTests(WriterSetup):
    def test_equality(self) -> None:
        correct_output = FromText.to_uniform_bytes
        output = self.writer._uniform_bytes(FromText.to_binary)
        self.assertArrayEqual(correct_output, output)

    def test_type(self) -> None:
        output = self.writer._uniform_bytes(FromText.to_binary)
        self.assertIsInstance(output, np.ndarray)

    def test_dtype(self) -> None:
        output = self.writer._uniform_bytes(FromText.to_binary).dtype.type
        self.assertIs(output, np.str_)


class SplitIntoColumnBitsTests(WriterSetup):
    def test_equality(self) -> None:
        correct_output = FromText.to_column_bits
        output = self.writer._split_into_column_bits(FromText.to_uniform_bytes)
        self.assertArrayEqual(output, correct_output)

    def test_type(self) -> None:
        output = self.writer._split_into_column_bits(FromText.to_uniform_bytes)
        self.assertIsInstance(output, np.ndarray)

    def test_dtype(self) -> None:
        output = self.writer._split_into_column_bits(FromText.to_uniform_bytes).dtype
        self.assertEqual(output, np.integer)


class ResizeTests(WriterSetup):
    def test_equality(self) -> None:
        writer = Writer(np.ones((5,)))
        correct_output = writer.img.shape
        output = writer._resize(np.ones((1,))).shape
        self.assertTupleEqual(correct_output, output)

    def test_type(self) -> None:
        output = self.writer._resize(np.ones((1,)))
        self.assertIsInstance(output, np.ndarray)

    def test_dtype(self) -> None:
        correct_output = self.writer.dtype
        output = self.writer._resize(np.ones((1,), dtype=correct_output)).dtype
        self.assertEqual(correct_output, output)


if __name__ == '__main__':
    unittest.main()
