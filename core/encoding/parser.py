import numpy as np

NUM_BITS = 8
DTYPE = f'uint{NUM_BITS}'


class Reader:
    def __init__(self, image: np.ndarray, dtype: str = DTYPE):
        self._img = image
        self.dtype = np.dtype(dtype)
 
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

    def read_hidden_bytes(self) -> np.ndarray:
        raise NotImplementedError

    def _group_bits(self) -> np.ndarray:
        bits = self._get_last_2_bits(self.img.flatten())
        return np.resize(bits, self.bytes_shape)

    @staticmethod
    def _get_last_2_bits(array: np.ndarray) -> np.ndarray:
        # https://stackoverflow.com/questions/60085470/efficient-way-of-extracting-the-last-two-digits-of-every-element-in-a-numpy-arra'
        return array & 3


class InvalidImageException(Exception):
    pass
