import os
import string
import numpy as np

from core.config import DEFAULT_CONFIG


class Constants:
    char_pool: str = string.ascii_uppercase + string.ascii_lowercase + string.digits + string.punctuation + 'áéíóúñ'
    long_text: str = char_pool * 1000
    large_size: int = 5000000
    big_img: np.ndarray = np.random.randint(0, 2**DEFAULT_CONFIG.num_bits, large_size)
    small_img: np.ndarray = np.arange(2**DEFAULT_CONFIG.num_bits)
    max_exec_seconds: float = 0.1
    dtype = DEFAULT_CONFIG.dtype


class FromText:
    text: str = 'aá&^'
    to_bytearray: bytearray = b'a\xc3\xa1&^'
    to_ints: np.ndarray = np.array([97, 195, 161, 38, 94])
    to_binary: np.ndarray = np.array(['0b1100001', '0b11000011', '0b10100001', '0b100110', '0b1011110'])
    to_uniform_bytes: np.ndarray = np.array(['01100001', '11000011', '10100001', '00100110', '01011110'])
    to_column_bits: np.ndarray = np.array([
        [1, 2, 0, 1], [3, 0, 0, 3], [2, 2, 0, 1], [0, 2, 1, 2], [1, 1, 3, 2]
    ])


class FromImg:
    img: np.ndarray = np.array([12, 113])  # ['0b1100', '0b1110001']
    num_bits = 8
    num_encoding_bits = 2  # 11 mask
    masked_last_bits: np.ndarray = np.array([12, 112])  # ['0b1100', '0b1110000']
    masked_first_bits: np.ndarray = np.array([0, 1])  # ['0b1100', '0b1110000']


class TestImg:
    _name: str = 'img'
    
    folder_path: str = os.path.dirname(os.path.abspath(__file__))
    png_path: str = os.path.join(folder_path, f'{_name}.png')
    jpeg_path: str = os.path.join(folder_path, f'{_name}.jpeg')
    write_path: str = os.path.join(folder_path, f'test_write_img.{DEFAULT_CONFIG.img_extension}')

