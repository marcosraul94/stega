import unittest

from core.writer import Writer
from core.reader import Reader
from core.converter import Converter
from test.values import Constants


class IntegrationTests(unittest.TestCase):
    def test_write_read(self):
        txt = Constants.char_pool
        converter = Converter()
        encoded_txt = converter.encode(txt)
        encoded_img = Writer(Constants.big_img).insert_bytes(encoded_txt)
        encoded_rebuilt_txt = Reader(encoded_img).read_hidden_bytes()
        decoded_rebuilt_txt = converter.decode(encoded_rebuilt_txt)
        self.assertIn(txt, decoded_rebuilt_txt)


if __name__ == '__main__':
    unittest.main()
