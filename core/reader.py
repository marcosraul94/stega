import numpy as np

from core.image import ImgProcessor


class Reader(ImgProcessor):
    def read_hidden_bytes(self) -> bytearray:
        filtered_last_bits = self._filter_last_bits(self.img)
        filtered_last_bits.resize(self.bytes_shape, refcheck=False)
        joined_bits = self._join_bits(filtered_last_bits)
        return bytearray(np.frombuffer(joined_bits, dtype=self.dtype))

    def _join_bits(self, array: np.ndarray) -> np.ndarray:
        # array = [ [0, 0, 0, 0], [0, 0, 0, 1], ... [3, 3, 3, 3] ]
        # output = [0, 1, ... 255]
        # for 8bits
        mapper_matrix = [64, 16, 4, 1]
        # might be improved by allocating a new matrix instead of astype, but this is cheap
        return np.dot(array, mapper_matrix).astype(self.dtype)

    def _filter_last_bits(self, matrix: np.ndarray) -> np.ndarray:
        return self._apply_mask(matrix, '1'*self.num_encoding_bits)
