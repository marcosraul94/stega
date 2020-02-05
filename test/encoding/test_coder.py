import string
import unittest
import datetime


from core.encoding.encoder import Coder


class EncodingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.coder = Coder()
        self.char_pool = string.printable + 'áéíóúñ'
        self.large_text = self.char_pool * 1000
        self.max_seconds = 0.1

    def test_decode_encode(self) -> None:
        encoded = self.coder.encode(self.char_pool)
        decoded = self.coder.decode(encoded)
        self.assertEqual(decoded, self.char_pool)

    def test_decode_speed(self) -> None:
        encoded_text = self.coder.encode(self.large_text)
        start = datetime.datetime.now()

        self.coder.decode(encoded_text)

        diff = datetime.datetime.now() - start
        self.assertLess(diff.total_seconds(), self.max_seconds)

    def test_encode_speed(self) -> None:
        start = datetime.datetime.now()

        self.coder.encode(self.large_text)

        diff = datetime.datetime.now() - start
        self.assertLess(diff.total_seconds(), self.max_seconds)


if __name__ == '__main__':
    unittest.main()
