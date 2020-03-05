import numpy as np

from core.image import ImgProcessor


class Writer(ImgProcessor):
    def insert_bytes(self, data: bytearray) -> np.ndarray:        
        encoded_data_as_img = self._prepare_input(data)
        masked_img = self._prepare_img()
        return (masked_img + encoded_data_as_img).astype(self.dtype)

    def _prepare_input(self, data: bytearray):
        data_split = self._split_into_ints(data)
        binary = self._to_binary(data_split)
        uniformed_bytes = self._uniform_bytes(binary)
        grouped_bits = self._split_into_column_bits(uniformed_bytes)
        resized_data = self._resize(grouped_bits)

        encoding_mask = '1' * self.num_encoding_bits
        return self._apply_mask(resized_data, encoding_mask).astype(self.dtype)

    def _prepare_img(self) -> np.ndarray:
        img_mask = '1'*(self.num_bits - self.num_encoding_bits) + '0'*self.num_encoding_bits
        return self._apply_mask(self.img, img_mask).astype(self.dtype)

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
        return np.array([bits.replace('0b', '').zfill(8) for bits in matrix])

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
    def _apply_mask(matrix: np.ndarray, binary_mask: str) -> np.ndarray:
        # binary mask = '11'
        # input [ [7, 12, ...], ]
        # intermediate repr [ ['0b111', '0b1100', ...], ]
        # masked intermediate repr [ ['0b100', '0b1100', ...], ]
        # output [ [4, 12, ...], ]
        return matrix & int(binary_mask, 2)
