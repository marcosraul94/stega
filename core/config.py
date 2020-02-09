import numpy as np

NUM_BYTES: int = 1
NUM_ENCODING_BITS: int = 2
ENCODING: str = 'utf-8'


class Config:
    def __init__(self, 
        num_bytes: int = NUM_BYTES, 
        num_encoding_bits: int = NUM_ENCODING_BITS,
        encoding: str = ENCODING):

        self.num_bytes = num_bytes
        self.num_encoding_bits = num_encoding_bits
        self.encoding = encoding


    @property
    def dtype(self) -> np.dtype:
        return np.dtype(f'uint{self.num_bits}')

    @property
    def num_columns(self) -> int:
        return self.num_bits // self.num_encoding_bits

    @property
    def num_bits(self) -> int:
        return self.num_bytes * 8
    
DEFAULT_CONFIG = Config()