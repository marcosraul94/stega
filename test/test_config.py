import unittest
import numpy as np

from core.config import Config


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config()

    def test_dtype_type(self) -> None:
        self.assertIsInstance(self.config.dtype, np.dtype)

    def test_dtype_value(self) -> None:
        self.config.num_bytes = 2
        self.assertEqual(self.config.dtype, np.uint16)

    def test_num_columns(self) -> None:
        self.config = Config(num_bytes = 4, num_encoding_bits = 4)
        self.assertEqual(self.config.num_columns, 8)

    def test_num_bits(self) -> None:
        self.config = Config(num_bytes = 4)
        self.assertEqual(self.config.num_bits, 32)


if __name__ == '__main__':
    unittest.main()


