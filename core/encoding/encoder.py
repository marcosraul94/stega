BYTE_ORDER: str = 'little'  # "little" the most significant byte is at the end of the byte array.
ENCODING: str = 'utf-8'  # "utf-8", 1-4 bytes to represent


class Coder:
    def __init__(self, encoding: str = ENCODING, byteorder: str = BYTE_ORDER):
        f"""
        Converts data between text and integers.
        @param encoding: Encoding scheme to convert a string into a bytes object. Default is '{ENCODING}'.
        @param byteorder: The byte order used to represent the integer. Default is '{BYTE_ORDER}'.
        """
        self.encoding = encoding
        self.byte_order = byteorder

    def encode(self, text: str) -> int:
        bytes_encoded = text.encode(self.encoding)
        int_repr = int.from_bytes(bytes_encoded, byteorder=self.byte_order)
        return int_repr

    def decode(self, integer: int) -> str:
        bytes_encoded = integer.to_bytes(length=(integer.bit_length() + 7) // 8, byteorder=self.byte_order)
        str_repr = bytes_encoded.decode(encoding=self.encoding)
        return str_repr
