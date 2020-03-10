import os
import cv2
import numpy as np

from core.reader import Reader
from core.writer import Writer
from core.config import Config
from core.config import DEFAULT_CONFIG
from core.converter import Converter


class StegaImage:
    def __init__(self, path, config: Config = DEFAULT_CONFIG):
        self.config = config
        self.load()

    def load(self, path) -> None:
        self.matrix = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    def save(self, output_path: str) -> None:
        cv2.imwrite(output_path, self.matrix)

    def write(self, txt: str) -> None:
        encoded_data = Converter(self.config.encoding).encode(txt)
        writer = Writer(
            self.matrix, self.config.dtype, self.config.num_columns,
            self.config.num_bits, self.num_encoding_bits
        )
        self.matrix = writer.insert_bytes(encoded_data)

    def read(self) -> str:
        reader = Reader(
            self.matrix, self.config.dtype, self.config.num_columns,
            self.config.num_bits, self.config.num_encoding_bits
        )
        encoded_data = reader.read_hidden_bytes()
        return Converter(self.config.encoding).decode(encoded_data) 
        