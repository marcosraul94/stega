import numpy as np
from core.reader import Reader
from core.config import DEFAULT_CONFIG


class Writer:
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
            # here we can set a decorator for validation and calling exit
        self._img = img.astype(self.dtype) if img.dtype != self.dtype else img

    @property
    def bytes_shape(self) -> tuple:
        return self.img.size // self.num_columns, self.num_columns

    def insert_bytes(self, data: bytearray) -> np.ndarray:
        # steps
        # 1 -> split the data in an int list
        # 2 -> convert each int into a binary repr
        # 3 -> split each binary int num columns last bits 
        # 4 -> resize that split to match image shape (data is ready to be added to the img)
        # 4 -> apply mask to image to clean last num of bits
        # 5 -> add image_masked and data_ready
        
        data_split = self._split_into_ints(data)
        binary = self._to_binary(data_split)
        grouped_bits = self._split_into_column_bits(binary)
        resized_data = self._resize(grouped_bits)
        
        encoding_mask = '1'*self.num_encoding_bits
        masked_data = self._apply_mask(resized_data, encoding_mask)
        
        img_mask = '1'*(self.num_bits - self.num_encoding_bits) + '0'*self.num_encoding_bits
        masked_img = self._apply_mask(self.img, img_mask)

        return masked_img + masked_data

    def _split_into_ints(self, data: bytearray) -> np.ndarray:
        return np.frombuffer(data, dtype=self.dtype)

    @staticmethod
    def _to_binary(ints: np.ndarray) -> np.ndarray:
        return np.array([bin(x) for x in ints.flatten()])

    def _split_into_column_bits(self, binary: np.ndarray):
        # input ['0b0', '0b10', '0b11', '0b100']
        


        extract_bits = lambda bits: int(bits.replace('b', '0'), 2)
        # not all ints take full 8 chars, use enumerate instead
        iterations = range(0, self.num_bits, self.num_encoding_bits)
        return np.array(list(
            extract_bits(bits[i:i+2]) for i in iterations for bits in binary
        ))

    def _resize(self, matrix: np.ndarray) -> np.ndarray:
        shallow_copy = np.copy(matrix)
        shallow_copy.resize(self.bytes_shape)
        return shallow_copy

    @staticmethod
    def _apply_mask(matrix: np.ndarray, binary_mask: str):
        return matrix & int(binary_mask, 2)
