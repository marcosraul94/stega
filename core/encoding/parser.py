import numpy as np

NUM_BITS = 8
DTYPE = f'uint{NUM_BITS}'


class Reader:
    def __init__(self, image: np.ndarray, dtype: str = DTYPE):
        self.dtype = np.dtype(dtype)
        self.img = image

    @property
    def img(self) -> np.ndarray:
        return self._img
           
    @img.setter
    def img(self, img: np.ndarray) -> None:
        if not img.size:
            raise InvalidImageException('Invalid image data.')

        self._img = img.astype(self.dtype) if img.dtype == self.dtype else img

    @property
    def bytes_shape(self) -> tuple:
        columns = int(NUM_BITS / 2)
        return self.img.size // columns, columns

    def read_hidden_bytes(self) -> int:
        grouped_bits = self._group_bits(self.img)
        joined_bits = self._join_bits(grouped_bits)
        return int(''.join(str(row) for row in joined_bits))

    @staticmethod
    def _join_bits(array: np.ndarray) -> np.ndarray:
        # array = [ [0, 0, 0, 0], [0, 0, 0, 1], ... [3, 3, 3, 3] ]
        # output = [0, 1, ... 255]
        mapper_matrix = [64, 16, 4, 1]
        return np.dot(array, mapper_matrix)

    def _group_bits(self, matrix: np.ndarray) -> np.ndarray:
        bits = self._get_last_2_bits(matrix.flatten())
        return np.resize(bits, self.bytes_shape)

    @staticmethod
    def _get_last_2_bits(array: np.ndarray) -> np.ndarray:
        # https://stackoverflow.com/questions/60085470/efficient-way-of-extracting-the-last-two-digits-of-every-element-in-a-numpy-arra'
        # array = [12, 55, 42, 0, ...]
        # output = [0, 11, 10, 11 ...]
        return array & 3


class InvalidImageException(Exception):
    pass
