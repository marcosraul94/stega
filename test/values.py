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


class Output:
    input_text: str = 'aá&^'
    to_bytearray: bytearray = b'a\xc3\xa1&^'
    to_ints: np.ndarray = np.array([97, 195, 161, 38, 94])
    to_binary: np.ndarray = np.array(['0b1100001', '0b11000011', '0b10100001', '0b100110', '0b1011110'])
    to_uniform_bytes: np.ndarray = np.array(['01100001', '11000011', '10100001', '00100110', '01011110'])
