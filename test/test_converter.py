import string
import unittest
from time import time

from core.converter import Converter


class ConverterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.converter = Converter()
        self.char_pool = string.printable + 'áéíóúñ'
        self.large_text = self.char_pool * 1000
        self.max_seconds = 0.1

    def test_decode_encode(self) -> None:
        encoded = self.converter.encode(self.char_pool)
        decoded = self.converter.decode(encoded)
        self.assertEqual(self.char_pool, decoded)

    def test_decode_type(self) -> None:
        encoded = self.converter.encode(self.char_pool)
        decoded = self.converter.decode(encoded)
        self.assertIsInstance(decoded, str)

    def test_encode_type(self) -> None:
        encoded = self.converter.encode(self.char_pool)
        self.assertIsInstance(encoded, bytes)

    def test_decode_speed(self) -> None:
        encoded_text = self.converter.encode(self.large_text)
        start = time()
        self.converter.decode(encoded_text)
        exec_seconds = time() - start
        self.assertLess(exec_seconds, self.max_seconds)

    def test_encode_speed(self) -> None:
        start = time()
        self.converter.encode(self.large_text)
        exec_seconds = time() - start
        self.assertLess(exec_seconds, self.max_seconds)


if __name__ == '__main__':
    unittest.main()
