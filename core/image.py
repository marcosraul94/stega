import numpy as np

from core.config import DEFAULT_CONFIG


class InvalidImageException(Exception):
    pass


class ImgProcessor:
    def __init__(self,
                 image: np.ndarray,
                 dtype: np.dtype = DEFAULT_CONFIG.dtype,
                 num_columns: int = DEFAULT_CONFIG.num_columns,
                 num_bits: int = DEFAULT_CONFIG.num_bits,
                 num_encoding_bits: int = DEFAULT_CONFIG.num_encoding_bits):
        self.dtype = dtype
        self.num_columns = num_columns
        self.img = image
        self.num_bits = num_bits
        self.num_encoding_bits = num_encoding_bits

    @property
    def img(self) -> np.ndarray:
        return self._img

    @img.setter
    def img(self, img: np.ndarray) -> None:
        if not img.size:
            raise InvalidImageException('Invalid image data.')
        self._img = img.astype(self.dtype) if img.dtype != self.dtype else img

    @property
    def bytes_shape(self) -> tuple:
        return self.img.size // self.num_columns, self.num_columns
