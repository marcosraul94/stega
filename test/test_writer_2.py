import unittest
import numpy as np

from core.writer import Writer
from core.reader import Reader
from test.values import Constants
from test.values import FromImg
from core.config import Config
from core.config import DEFAULT_CONFIG
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
        self.assertEqual(output, correct_output)

    def test_diff_dtype(self) -> None:
        correct_output = np.dtype('uint32')
        diff_dtype = np.dtype('uint8')
        img = self.constants.small_img.astype(diff_dtype)
        writer = Writer(img, dtype=correct_output)
        output = writer.dtype
        self.assertEqual(output, correct_output)


class BytesShapeTests(WriterSetup):
    def test_equality(self) -> None:
        size = 5
        num_columns = 4
        correct_output = (1, num_columns)
        output = Writer(np.arange(size)).bytes_shape
        self.assertTupleEqual(output, correct_output)

    def test_type(self) -> None:
        self.assertIsInstance(self.writer.bytes_shape, tuple)

    def test_len(self) -> None:
        correct_output = 2
        output = len(self.writer.bytes_shape)
        self.assertEqual(output, correct_output)


class InsertBytesTests(WriterSetup):
    def test_equality(self) -> None:
        self.assertFalse(True)

    def test_type(self) -> None:
        self.assertFalse(True)

    def test_dtype(self) -> None:
        self.assertFalse(True)


class PrepareInputTests(WriterSetup):
    def test_equality(self) -> None:
        self.assertFalse(True)

    def test_type(self) -> None:
        self.assertFalse(True)

    def test_dtype(self) -> None:
        self.assertFalse(True)


class PrepareImgTests(WriterSetup):
    def test_equality(self) -> None:
        writer = Writer(
            self.from_img.img, num_bits=self.from_img.num_bits, num_encoding_bits=self.from_img.num_encoding_bits
        )
        correct_output = self.from_img.prepared
        output = writer._prepare_img()
        self.assertArrayEqual(output, correct_output)

    def test_type(self) -> None:
        output = self.writer._prepare_img()
        self.assertIsInstance(output, np.ndarray)

    def test_dtype(self) -> None:
        correct_output = self.writer.dtype
        output = self.writer._prepare_img().dtype
        self.assertEqual(output, correct_output)


class SplitIntoIntsTests(WriterSetup):
    def test_equality(self) -> None:
        self.assertFalse(True)

    def test_type(self) -> None:
        self.assertFalse(True)

    def test_dtype(self) -> None:
        self.assertFalse(False)




class ResizeTests(WriterSetup):
    def test_equality(self) -> None:
        self.assertFalse(True)

    def test_type(self) -> None:
        self.assertFalse(True)

    def test_dtype(self) -> None:
        self.assertFalse(True)


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
