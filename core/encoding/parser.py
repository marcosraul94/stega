import cv2
import numpy as np
from operator import add

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
        return self.img.size // (NUM_BITS/2), NUM_BITS/2

    def read_hidden_bytes(self) -> np.ndarray:
        from datetime import datetime
        grouped_bits = self._group_bits()

        start = datetime.now()
        joined_bytes =  np.array([
            int(''.join(bits), 2) for bits in grouped_bits
        ])
        diff = (datetime.now() - start).total_seconds()
        print('joined_bytes slow:', diff)
        print(grouped_bits.shape)

        start = datetime.now()
        f = (int(''.join(bits), 2) for bits in grouped_bits)
        v2 = np.fromiter(f)
        diff = (datetime.now() - start).total_seconds()
        print('joined_bytes v2:', diff)

        return joined_bytes

    def _group_bits(self) -> np.ndarray:
        bits = self._get_last_2_bits(self.img.flatten())
        return np.resize(bits, self.bytes_shape)

    def _get_last_2_bits(self, array: np.ndarray) -> np.ndarray:
        # https://stackoverflow.com/questions/60085470/efficient-way-of-extracting-the-last-two-digits-of-every-element-in-a-numpy-arra'
        bits_map = np.array(['00', '01', '10', '11'])
        return bits_map[array & 3]



class InvalidImageException(Exception):
    pass