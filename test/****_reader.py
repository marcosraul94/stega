import unittest
import numpy as np
import itertools
from time import time

from core.reader import Reader
from core.converter import Converter
from core.reader import InvalidImageException
from test.values import BIG_IMG
from test.values import SMALL_IMG
from test.values import LARGE_SIZE
from test.values import MAX_EXEC_SECONDS


class ReaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reader = Reader(SMALL_IMG)
        self.max_seconds = MAX_EXEC_SECONDS
        self.large_size = LARGE_SIZE
        self.big_img = BIG_IMG

    def test_img_type(self) -> None:
        self.assertIsInstance(self.reader.img, np.ndarray)

    def test_invalid_img(self) -> None:
        self.assertRaises(InvalidImageException, lambda: Reader(np.array([])))

    def test_img_dtype(self) -> None:
        wrong_dtype = np.uint16
        correct_dtype = np.uint8
        img = np.arange(10).astype(wrong_dtype)
        reader = Reader(img, dtype = correct_dtype)
        self.assertEqual(reader.img.dtype, correct_dtype)

    def test_bytes_shape_type(self) -> None:
        self.assertIsInstance(self.reader.bytes_shape, tuple)

    def test_bytes_shape_len(self) -> None:
        self.assertEqual(len(self.reader.bytes_shape), 2)

    def test_bytes_shape_return_types(self) -> None:
        self.assertTrue(all(type(x) == int for x in self.reader.bytes_shape))

    def test_bytes_shape_correct_modulus(self) -> None:
        self.reader = Reader(np.arange(self.reader.num_columns))
        size = np.resize(self.reader.img, self.reader.bytes_shape).size
        dividable_by_num_columns = (size % self.reader.num_columns) == 0
        self.assertTrue(dividable_by_num_columns)

    def test_bytes_shape_incorrect_modulus(self) -> None:
        self.reader = Reader(np.arange(self.reader.num_columns + 1))
        size = np.resize(self.reader.img, self.reader.bytes_shape).size
        dividable_by_num_columns = (size % self.reader.num_columns) == 0
        self.assertTrue(dividable_by_num_columns)

    def test_read_hidden_bytes(self) -> None:
        txt = 'á'
        encoding = 'utf-8'
        converter = Converter(encoding)
        encoded = converter.encode(txt)  # b'\xc3\xa1'
        # writer mock
        ints = np.frombuffer(encoded, 'uint8')  # [195, 161]
        bins = [bin(x).replace('0b', '').zfill(8) for x in ints]  # ['11000011', '10100001']
        split_bins = [['11', '00', '00', '11'], ['10', '10', '00', '01']]
        split_dec = np.array([[3, 0, 0, 3], [2, 2, 0, 1]]).flatten()
        masked_img = np.arange(8).astype('uint8') & int('11111100', 2)
        encoded_img = masked_img + split_dec
        # end writer mock
        encoded_bytes = Reader(encoded_img).read_hidden_bytes()
        output = converter.decode(encoded_bytes)
        self.assertEqual(output, txt)

    def test_read_hidden_bytes_type(self) -> None:
        hidden_bytes = self.reader.read_hidden_bytes()
        self.assertIsInstance(hidden_bytes, bytearray)

    def test_read_hidden_bytes_speed(self) -> None:
        self.reader = Reader(self.big_img)
        start = time()
        self.reader.read_hidden_bytes()
        exec_seconds = time() - start
        self.assertLess(exec_seconds, self.max_seconds)

    def test_join_bits(self) -> None:
        columns = self.reader.num_columns
        possible_values = [x for _ in range(columns) for x in range(columns)]
        matrix = np.array(sorted(set(x for x in itertools.permutations(possible_values, columns))))

        correct_output = np.array([x for x in range(len(matrix))])
        output = self.reader._join_bits(matrix)
        self.assertTrue(np.array_equal(output, correct_output))

    def test_join_bits_type(self) -> None:
        matrix = np.resize(self.reader.img, self.reader.bytes_shape).astype(self.reader.dtype)
        output = self.reader._join_bits(matrix)
        self.assertIsInstance(output, np.ndarray)

    def test_join_bits_dtype(self) -> None:
        matrix = np.resize(self.reader.img, self.reader.bytes_shape).astype(self.reader.dtype)
        output = self.reader._join_bits(matrix)
        self.assertEqual(output.dtype, self.reader.dtype)

    def test_join_bits_speed(self) -> None:
        reader = Reader(self.big_img)
        large_matrix = np.resize(reader.img, reader.bytes_shape).astype(reader.dtype)
        start = time()
        self.reader._join_bits(large_matrix)
        exec_seconds = time() - start
        self.assertLess(exec_seconds, self.max_seconds)

    def test_group_bits(self) -> None:
        self.reader.img = np.array([
            np.array([0, 1, 2, 3, 4]),
        ])
        correct_output = np.array([
            np.array([0, 1, 2, 3]),
        ])
        output = self.reader._group_bits(self.reader.img)
        self.assertTrue(np.array_equal(output, correct_output))

    def test_group_bits_type(self) -> None:
        output = self.reader._group_bits(self.reader.img)
        self.assertIsInstance(output, np.ndarray)

    def test_group_bits_dtype(self) -> None:
        output = self.reader._group_bits(self.reader.img)
        self.assertEqual(output.dtype, self.reader.dtype)

    def test_group_bits_speed(self) -> None:
        reader = Reader(self.big_img)
        start = time()
        reader._group_bits(reader.img)
        exec_seconds = time() - start
        self.assertLess(exec_seconds, self.max_seconds)

    def test_get_last_2_bits(self) -> None:
        correct_output = np.array([int(bin(x)[-2:].replace('b', '0'), 2) for x in self.reader.img])
        output = self.reader._get_last_bits(self.reader.img.flatten())
        self.assertTrue(np.array_equal(output, correct_output))


if __name__ == '__main__':
    unittest.main()
