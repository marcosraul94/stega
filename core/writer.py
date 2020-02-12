import numpy as np
from core.config import DEFAULT_CONFIG
from core.image import InvalidImageException


# rewrite this with inheritance and config as an entire object? Well think about the last one.
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
        uniformed_bytes = self._uniform_bytes(binary)
        grouped_bits = self._split_into_column_bits(uniformed_bytes)
        resized_data = self._resize(grouped_bits)
        
        encoding_mask = '1'*self.num_encoding_bits
        masked_data = self._apply_mask(resized_data, encoding_mask)
        
        img_mask = '1'*(self.num_bits - self.num_encoding_bits) + '0'*self.num_encoding_bits
        masked_img = self._apply_mask(self.img, img_mask)

        return masked_img + masked_data

    def _prepare_input(self, data: bytearray):
        data_split = self._split_into_ints(data)
        binary = self._to_binary(data_split)
        uniformed_bytes = self._uniform_bytes(binary)
        grouped_bits = self._split_into_column_bits(uniformed_bytes)
        resized_data = self._resize(grouped_bits)

        encoding_mask = '1' * self.num_encoding_bits
        return self._apply_mask(resized_data, encoding_mask)

    def _prepare_img(self, img: np.ndarray):
        img_mask = '1'*(self.num_bits - self.num_encoding_bits) + '0'*self.num_encoding_bits
        return self._apply_mask(img, img_mask)

    def _split_into_ints(self, data: bytearray) -> np.ndarray:
        # input b'aeio\xc3\xba'
        # output [97, 101, 105, 111, 195, 186]
        return np.frombuffer(data, dtype=self.dtype)

    @staticmethod
    def _to_binary(ints: np.ndarray) -> np.ndarray:
        return np.array([bin(x) for x in ints.flatten()])

    @staticmethod
    def _uniform_bytes(matrix: np.ndarray) -> np.ndarray:
        # input ['0b100', ]
        # output [ '00000100', ]
        return np.array([bits.replace('b', '0').zfill(8) for bits in matrix])

    @staticmethod
    def _split_into_column_bits(matrix: np.ndarray) -> np.ndarray:
        # input ['00000100', ]
        # output [ [0, 0, 1, 0], ]
        return np.array([
            np.array([
                int(bits[i*2: (i+1)*2], 2) for i in range(4)
            ]) for bits in matrix
        ])

    def _resize(self, matrix: np.ndarray) -> np.ndarray:
        shallow_copy = np.copy(matrix)
        shallow_copy.resize(self.img.shape, refcheck=False)  # not the same np.resize and .resize
        return shallow_copy

    @staticmethod
    def _apply_mask(matrix: np.ndarray, binary_mask: str):
        # binary mask = '11'
        # input [ [7, 12, ...], ]
        # intermediate repr [ ['0b111', '0b1100', ...], ]
        # masked intermediate repr [ ['0b100', '0b1100', ...], ]
        # output [ [4, 12, ...], ]
        return matrix & int(binary_mask, 2)
