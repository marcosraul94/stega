import os
import unittest
import numpy as np

from core.stegapy import Stegapy
from test.values import TestImg
from test.utils import TestCaseWithArrayEqual



class StegapySetup(TestCaseWithArrayEqual):
    def tearDown(self) -> None:
        if os.path.isfile(TestImg.write_path):
            os.remove(TestImg.write_path)

class ReadWriteImageTests(StegapySetup):
    def test_png_img(self):
        self._test_read_write_equality(TestImg.png_path)

    def test_jpeg_img(self):
        self._test_read_write_equality(TestImg.jpeg_path)

    def _test_read_write_equality(self, source_img_path: str) -> None:
        stegapy = Stegapy(source_img_path)
        correct_output = stegapy._read()
        
        stegapy._write(correct_output, TestImg.write_path)
        output = Stegapy(TestImg.write_path)._read()
        self.assertArrayEqual(correct_output, output)


if __name__ == '__main__':
    unittest.main()
