from core.config import DEFAULT_CONFIG


class Converter:
    def __init__(self, encoding: str = DEFAULT_CONFIG.encoding):
        """
        Converts data between text and integers.
        @param encoding: Encoding scheme to convert a string into a bytes object and vice versa."""
        f"""Default is '{DEFAULT_CONFIG.encoding}'.
        """
        self.encoding = encoding

    def encode(self, text: str) -> bytearray:
        return bytearray(text.encode(self.encoding))

    def decode(self, bytes_encoded: bytearray) -> str:
        return bytes_encoded.decode(encoding=self.encoding)
