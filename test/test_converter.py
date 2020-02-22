import unittest
from time import time

from core.converter import Converter
from test.values import Constants

class ConverterSetup(unittest.TestCase):
    def setUp(self) -> None:
        self.converter = Converter()
        self.constants = Constants


class EncodeDecodeTests(ConverterSetup):
    def test_equality(self) -> None:
        encoded = self.converter.encode(self.constants.char_pool)
        decoded = self.converter.decode(encoded)
        self.assertEqual(decoded, self.constants.char_pool)

    def test_speed(self) -> None:
        start = time()
        encoded = self.converter.encode(self.constants.long_text)
        self.converter.decode(encoded)
        exec_seconds = time() - start
        self.assertLess(exec_seconds, self.constants.max_exec_seconds)


class EncodeTests(ConverterSetup):
    def test_type(self) -> None:
        encoded = self.converter.encode(self.constants.char_pool)
        self.assertIsInstance(encoded, bytearray)

    def test_speed(self) -> None:
        start = time()
        self.converter.encode(self.constants.long_text)
        exec_seconds = time() - start
        self.assertLess(exec_seconds, self.constants.max_exec_seconds)


class DecodeTests(ConverterSetup):
    def test_type(self) -> None:
        encoded = self.converter.encode(self.constants.char_pool)
        decoded = self.converter.decode(encoded)
        self.assertIsInstance(decoded, str)

    def test_decode_speed(self) -> None:
        encoded_text = self.converter.encode(self.constants.long_text)
        start = time()
        self.converter.decode(encoded_text)
        exec_seconds = time() - start
        self.assertLess(exec_seconds, self.constants.max_exec_seconds)


if __name__ == '__main__':
    unittest.main()
