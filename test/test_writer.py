import unittest
import numpy as np

from core.writer import Writer
from core.reader import Reader
from test.values import Constants
from test.values import FromImg
from test.values import FromText
from core.converter import Converter
from core.image import InvalidImageException


class WriterSetup(unittest.TestCase):
    def setUp(self) -> None:
        self.constants = Constants
        self.from_img = FromImg
        self.writer = Writer(self.constants.small_img)
        self.converter = Converter()

    def assertArrayEqual(self, first: np.ndarray, second: np.ndarray, msg: str = None) -> None:
        if not np.array_equal(first, second):
            self.fail(f'{first} not equal to {second}' + f'{": " + msg if msg else ""}')


class ImgTests(WriterSetup):
    def test_invalid_img(self) -> None:
        self.assertRaises(InvalidImageException, lambda: Writer(np.array([])))

    def test_type(self) -> None:
        self.assertIsInstance(self.writer.img, np.ndarray)

    def test_same_dtype(self) -> None:
        correct_output = np.dtype('uint32')
        img = self.constants.small_img.astype(correct_output)
        writer = Writer(img, dtype=correct_output)
        output = writer.dtype
        self.assertEqual(correct_output, output)

    def test_diff_dtype(self) -> None:
        correct_output = np.dtype('uint32')
        diff_dtype = np.dtype('uint8')
        img = self.constants.small_img.astype(diff_dtype)
        writer = Writer(img, dtype=correct_output)
        output = writer.dtype
        self.assertEqual(correct_output, output)


class BytesShapeTests(WriterSetup):
    def test_equality(self) -> None:
        size = 5
        num_columns = 4
        correct_output = (1, num_columns)
        output = Writer(np.arange(size)).bytes_shape
        self.assertTupleEqual(correct_output, output)

    def test_type(self) -> None:
        self.assertIsInstance(self.writer.bytes_shape, tuple)

    def test_len(self) -> None:
        correct_output = 2
        output = len(self.writer.bytes_shape)
        self.assertEqual(correct_output, output)


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
            self.from_img.img, num_bits=self.from_img.num_bits, num_encoding_bits=self.from_img.num_encoding_bits
        )
        correct_output = self.from_img.prepared
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


class ApplyMaskTests(WriterSetup):
    def test_equality(self) -> None:
        img = np.array([7, 14])  # ['0b111', '0b1110']
        output = self.writer._apply_mask(img, '11111011')
        correct_output = np.array([3, 10])  # ['0b11', '0b1010']
        self.assertArrayEqual(output, correct_output)

    def test_type(self) -> None:
        output = self.writer._apply_mask(self.writer.img, '0')
        self.assertIsInstance(output, np.ndarray)

    def test_dtype(self) -> None:
        correct_output = self.writer.dtype
        output = self.writer._apply_mask(self.writer.img, '0').dtype
        self.assertEqual(output, correct_output)


if __name__ == '__main__':
    unittest.main()
