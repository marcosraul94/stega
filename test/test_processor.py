import unittest
import numpy as np

from core.image import ImgProcessor
from core.image import InvalidImageException
from test.values import Constants
from test.utils import TestCaseWithArrayEqual


class ProcessorSetup(TestCaseWithArrayEqual):
    def setUp(self) -> None:
        self.img_processor = ImgProcessor(Constants.small_img)


class ImgTests(ProcessorSetup):
    def test_equality(self) -> None:
        correct_output = np.dtype('uint32')
        diff_dtype = np.dtype('uint8')
        img = Constants.small_img.astype(diff_dtype)
        img_processor = ImgProcessor(img, dtype=correct_output)
        output = img_processor.dtype
        self.assertEqual(correct_output, output)

    def test_invalid_img(self) -> None:
        self.assertRaises(InvalidImageException, lambda: ImgProcessor(np.array([])))

    def test_type(self) -> None:
        self.assertIsInstance(self.img_processor.img, np.ndarray)


class BytesShapeTests(ProcessorSetup):
    def test_equality(self) -> None:
        size = 5
        num_columns = 4
        correct_output = (1, num_columns)
        output = ImgProcessor(np.arange(size)).bytes_shape
        self.assertTupleEqual(correct_output, output)

    def test_type(self) -> None:
        self.assertIsInstance(self.img_processor.bytes_shape, tuple)

    def test_len(self) -> None:
        correct_output = 2
        output = len(self.img_processor.bytes_shape)
        self.assertEqual(correct_output, output)


if __name__ == '__main__':
    unittest.main()
