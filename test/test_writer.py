import unittest
import numpy as np

from core.config import Config
from core.writer import Writer
from core.reader import Reader
from core.config import DEFAULT_CONFIG
from core.converter import Converter
from test.values import CHAR_POOL
from test.values import BIG_IMG
from test.values import SMALL_IMG
from test.values import LARGE_SIZE
from test.values import MAX_EXEC_SECONDS


class WriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.writer = Writer(SMALL_IMG)
        self.converter = Converter()
        self.char_pool = CHAR_POOL
        self.max_seconds = MAX_EXEC_SECONDS
        self.large_size = LARGE_SIZE
        self.big_img = BIG_IMG

    def test_insert_bytes(self) -> None:
        converter = Converter()
        encoded = converter.encode(self.char_pool)
        encoded_img = Writer(np.arange(len(self.char_pool)*4)).insert_bytes(encoded)
        reconstructed_encoded_info = Reader(encoded_img).read_hidden_bytes()
        reconstructed_decoded_info = converter.decode(reconstructed_encoded_info)
        self.assertEqual(reconstructed_decoded_info, self.char_pool)

    def test_split_into_ints(self) -> None:
        encoded = self.converter.encode(self.char_pool)
        correct_output = np.array(list(encoded))
        output = self.writer._split_into_ints(encoded)
        self.assertTrue(np.array_equal(output, correct_output))

    def test_split_into_ints_type(self) -> None:
        encoded = self.converter.encode(self.char_pool)
        output = self.writer._split_into_ints(encoded)
        self.assertIsInstance(output, np.ndarray)

    def test_split_into_ints_dtype(self) -> None:
        encoded = self.converter.encode(self.char_pool)
        output = self.writer._split_into_ints(encoded)
        self.assertEqual(output.dtype, DEFAULT_CONFIG.dtype)

    def test_to_binary(self) -> None:
        ints = np.array([1, 2, 4])
        correct_output = np.array(['0b1', '0b10', '0b100'])
        output = self.writer._to_binary(ints)
        self.assertTrue(np.array_equal(output, correct_output))

    def test_to_binary_type(self) -> None:
        output = self.writer._to_binary(self.writer.img)
        self.assertIsInstance(output, np.ndarray)

    def test_to_binary_dtype(self) -> None:
        output = self.writer._to_binary(self.writer.img)
        is_bin = lambda x: x[:2] == '0b' and all(bit in '10' for bit in x[2:])
        self.assertTrue(all(is_bin(bits) for bits in output))

    def test_uniform_bytes(self) -> None:
        matrix = np.array(['0b100', ])
        correct_output = np.array(['00000100', ])
        output = self.writer._uniform_bytes(matrix)
        self.assertTrue(np.array_equal(output, correct_output))

    def test_split_into_column_bits(self) -> None:
        img = np.array(['00000001', '00000010', '00000011', '00000100'])
        correct_output = [
            [0, 0, 0, 1],
            [0, 0, 0, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 0]
        ]
        output = self.writer._split_into_column_bits(img)
        self.assertTrue(np.array_equal(output, correct_output))

    def test_split_into_column_bits_type(self) -> None:
        img = np.array(['00000001', ])
        output = self.writer._split_into_column_bits(img)
        self.assertIsInstance(output, np.ndarray)

    def test_resize_shape(self) -> None:
        size = 10
        config = Config(num_bytes=1, num_encoding_bits=2)
        img = np.arange(size)
        writer = Writer(img, num_encoding_bits=config.num_encoding_bits, num_bits=config.num_bits)
        correct_shape = img.shape
        output_shape = writer._resize(np.arange(size - 1)).shape
        self.assertTupleEqual(output_shape, correct_shape)

    def test_resize_add_missing_zeros(self) -> None:
        size = 10
        diff = size - 5
        config = Config(num_bytes=1, num_encoding_bits=2)
        img = np.arange(size)
        writer = Writer(img, num_encoding_bits=config.num_encoding_bits, num_bits=config.num_bits)
        zeros = writer._resize(np.arange(diff))[diff:]
        self.assertTrue(np.array_equal(np.zeros((size-diff,)), zeros))

    def test_apply_mask(self) -> None:
        matrix = np.array([255])
        writer = Writer(matrix, num_bits=8, num_encoding_bits=2)
        correct_output = np.array([int(bin(value).replace('0b', '')[:-2] + '00', 2) for value in matrix])
        output = writer._apply_mask(matrix, '1'*6 + '00')
        self.assertTrue(np.array_equal(output, correct_output))

    def test_apply_mask_type(self) -> None:
        output = self.writer._apply_mask(self.writer.img, '1')
        self.assertIsInstance(output, np.ndarray)

    def test_apply_mask_dtype(self) -> None:
        matrix = np.arange(10).astype(DEFAULT_CONFIG.dtype)
        output = self.writer._apply_mask(matrix, '1').dtype
        correct_output = DEFAULT_CONFIG.dtype
        self.assertEqual(output, correct_output)

    def test_prepare_input(self) -> None:
        in_bytearray = Converter().encode('aá&^')  # b'a\xc3\xa1&^'
        in_ints = np.frombuffer(in_bytearray, dtype=self.writer.dtype)  # [97, 195, 161, 38, 94]
        in_binary = [bin(x) for x in in_ints]  # ['01100001', '11000011', '10100001', '00100110', '01011110']
        # ['01', '10', '00', '01'], ['11', '00', '00', '11'], ['10', '10', '00', '01'], ['00', '10', '01', '10'], ['01', '01', '11', '10']
        img = np.arange(20).reshape(5, 4)
        # img ['0b0', '0b1', '0b10', '0b11', '0b100', '0b101', '0b110', '0b111', '0b1000', '0b1001', '0b1010', '0b1011', '0b1100', '0b1101', '0b1110', '0b1111', '0b10000', '0b10001', '0b10010', '0b10011']
        # masked [ 0  0  0  0  4  4  4  4  8  8  8  8 12 12 12 12 16 16 16 16]
        masked_img = img & int('11111100', 2)
        masked_data = np.array([[1, 2, 0, 1], [3, 0, 0, 3], [2, 2, 0, 1], [0, 2, 1, 2], [1, 1, 3, 2]])
        correct_output = masked_data + masked_img
        output = Writer(img).insert_bytes(in_bytearray)
        print(output)
        print(correct_output)
        self.assertTrue(np.array_equal(output, correct_output))


    def test_prepare_input_shape(self) -> None:
        shape = self.writer._prepare_input(self.writer.img).shape
        correct_shape = self.writer.img.shape
        self.assertEqual(shape, correct_shape)

    def test_prepare_input_type(self) -> None:
        output = self.writer._prepare_input(self.writer.img)
        self.assertIsInstance(output, np.ndarray)

    def test_prepare_input_dtype(self) -> None:
        output_dtype = self.writer._prepare_input(self.writer.img).dtype
        correct_dtype = self.writer.dtype
        self.assertEqual(output_dtype, correct_dtype)

    def test_prepare_img(self) -> None:
        matrix = np.array([255, 124])  # [11111111, 01111100]
        writer = Writer(matrix, num_bits=8, num_encoding_bits=2)  # mask 11111100
        correct_output = np.array([252, 124])  # [11111100, 01111100]
        output = writer._prepare_img(matrix)
        self.assertTrue(np.array_equal(output, correct_output))

    def test_prepare_img_shape(self) -> None:
        output_shape = self.writer._prepare_img(self.writer.img).shape
        correct_shape = self.writer.img.shape
        self.assertEqual(output_shape, correct_shape)

    def test_prepare_img_type(self) -> None:
        output = self.writer._prepare_img(self.writer.img)
        self.assertIsInstance(output, np.ndarray)

    def test_prepare_img_dtype(self) -> None:
        output_dtype = self.writer._prepare_img(self.writer.img)
        correct_dtype = self.writer.img.dtype


if __name__ == '__main__':
    unittest.main()
