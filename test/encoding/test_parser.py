import unittest
import numpy as np
import itertools

from core.encoding.parser import NUM_BITS
from core.encoding.parser import Reader
from core.encoding.parser import InvalidImageException


class ParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reader = Reader(np.arange(2 ** NUM_BITS))

    def test_get_last_2_bits(self) -> None:
        correct_output = np.array([int(bin(x)[-2:].replace('b', '0'), 2) for x in self.reader.img])
        output = self.reader._get_last_2_bits(self.reader.img.flatten())
        self.assertTrue(np.array_equal(correct_output, output))

    def test_invalid_img(self) -> None:
        self.assertRaises(InvalidImageException, lambda: Reader(np.array([])))

    def test_bytes_shape_correct_modulus(self) -> None:
        self.reader.img = np.arange(4)
        self.assertFalse(
            np.resize(self.reader.img, self.reader.bytes_shape).size % 4
        )

    def test_bytes_shape_incorrect_modulus(self) -> None:
        self.reader.img = np.arange(5)
        self.assertFalse(
            np.resize(self.reader.img, self.reader.bytes_shape).size % 4
        )

    def test_group_bits(self) -> None:
        self.reader.img = np.array([
            np.array([0, 1, 2, 3, 4]),
        ])
        correct_output = np.array([
            np.array([0, 1, 2, 3]),
        ])
        output = self.reader._group_bits(self.reader.img)
        self.assertTrue(np.array_equal(correct_output, output))

    def test_join_bits(self) -> None:
        columns = int(NUM_BITS / 2)
        possible_values = [x for _ in range(columns) for x in range(columns)]
        sample = np.array(sorted(set(x for x in itertools.permutations(possible_values, columns))))

        correct_output = np.array([x for x in range(len(sample))])
        output = self.reader._join_bits(sample)
        self.assertTrue(np.array_equal(correct_output, output))

    # def test_read_hidden_bytes(self) -> None:
    #     self.reader.img = self.img_sample
    #     correct_output = np.array([
    #         int(''.join(row), 2) for row in self.reader._group_bits()
    #     ])
    #     output = self.reader.read_hidden_bytes()
    #     self.assertTrue(np.array_equal(correct_output, output))
    #
    # def test_read_hidden_bytes_speed(self) -> None:
    #     self.reader.img = np.random.randint(0, 256, 5000000)
    #     start = datetime.datetime.now()
    #     self.reader.read_hidden_bytes()
    #     diff = datetime.datetime.now() - start
    #     self.assertLess(diff.total_seconds(), 4)


if __name__ == '__main__':
    unittest.main()
