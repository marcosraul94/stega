import cv2
import numpy as np


img_path = '/Users/intellisys/study/stega/test_img.jpg'


class Reader:
    def __init__(self, img_path: str):
        self.img = cv2.imread(img_path)


def get_bits(x: int) -> str:
    return str(bin(x))[-2:]


if __name__ == '__main__':
    reader = Reader(img_path)
    img = reader.img
    pixels = img.flatten()
    from datetime import datetime
    now = datetime.now()
    m1 = np.array(bin(x).zfill(2)[-2:] for x in pixels)
    print((datetime.now() - now).total_seconds())

    now = datetime.now()
    converter = np.vectorize(lambda x: np.binary_repr[-2:].replace('b', '0'))
    m2 = converter(pixels)
    print((datetime.now() - now).total_seconds())

    a = 0
    converted = bin(a)[-2:].replace('b', '0')
    print(converted)
