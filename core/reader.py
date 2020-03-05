import numpy as np

from core.image import ImgProcessor


class Reader(ImgProcessor):
    def read_hidden_bytes(self) -> bytearray:
        grouped_bits = self._group_bits(self.img)
        joined_bits = self._join_bits(grouped_bits)
        return bytearray(np.frombuffer(joined_bits, dtype=self.dtype))

    def _join_bits(self, array: np.ndarray) -> np.ndarray:
        # array = [ [0, 0, 0, 0], [0, 0, 0, 1], ... [3, 3, 3, 3] ]
        # output = [0, 1, ... 255]
        # for 8bits
        mapper_matrix = [64, 16, 4, 1]
        # might be improved by allocating a new matrix instead of astype, but this is cheap
        return np.dot(array, mapper_matrix).astype(self.dtype)

    def _group_bits(self, matrix: np.ndarray) -> np.ndarray:
        bits = self._get_last_bits(matrix.flatten())
        return np.resize(bits, self.bytes_shape)

    def _get_last_bits(self, matrix: np.ndarray) -> np.ndarray:
        # https://stackoverflow.com/questions/60085470/efficient-way-of-extracting-the-last-two-digits-of-every-element-in-a-numpy-arra'
        # array = [12, 55, 42, 0, ...]
        # output = [0, 11, 10, 11 ...]
        return matrix & int('1'*self.num_encoding_bits, 2)
