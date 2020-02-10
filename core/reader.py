import numpy as np

from core.config import DEFAULT_CONFIG
from core.image import InvalidImageException


class Reader:
    def __init__(self,
                 image: np.ndarray,
                 dtype: np.dtype = DEFAULT_CONFIG.dtype,
                 num_columns: int = DEFAULT_CONFIG.num_columns):

        self.dtype = dtype
        self.num_columns = num_columns
        self.img = image

    @property
    def img(self) -> np.ndarray:
        return self._img
           
    @img.setter
    def img(self, img: np.ndarray) -> None:
        if not img.size:
            raise InvalidImageException('Invalid image data.')
            # here we can set a decorator for validation and calling exit
        self._img = img.astype(self.dtype) if img.dtype != self.dtype else img

    @property
    def bytes_shape(self) -> tuple:
        return self.img.size // self.num_columns, self.num_columns

    def read_hidden_bytes(self) -> bytearray:
        # needs to be tested
        grouped_bits = self._group_bits(self.img)
        joined_bits = self._join_bits(grouped_bits)
        return bytearray(np.frombuffer(joined_bits, dtype=self.dtype))

    def _join_bits(self, array: np.ndarray) -> np.ndarray:
        # array = [ [0, 0, 0, 0], [0, 0, 0, 1], ... [3, 3, 3, 3] ]
        # output = [0, 1, ... 255]
        mapper_matrix = [64, 16, 4, 1]
        # might be improved by allocating a new matrix instead of astype, but this is cheap
        return np.dot(array, mapper_matrix).astype(self.dtype)

    def _group_bits(self, matrix: np.ndarray) -> np.ndarray:
        bits = self._get_last_2_bits(matrix.flatten())
        return np.resize(bits, self.bytes_shape)

    @staticmethod
    def _get_last_2_bits(matrix: np.ndarray) -> np.ndarray:
        # https://stackoverflow.com/questions/60085470/efficient-way-of-extracting-the-last-two-digits-of-every-element-in-a-numpy-arra'
        # array = [12, 55, 42, 0, ...]
        # output = [0, 11, 10, 11 ...]
        return matrix & 3
