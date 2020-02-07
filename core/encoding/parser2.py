import cv2
import numpy as np


img_path = 'test_img.jpg'


class Reader:
    def __init__(self, img_path: str):
        self.img = cv2.imread(img_path).astype('uint8')

def get_bits(x: int) -> str:
    return str(bin(x))[-2:]


if __name__ == '__main__':
    reader = Reader(img_path)
    img = reader.img
    pixels = img.flatten()

    from datetime import datetime
    def get_last_two_bits(arr_in):
        return bits_map[arr_in % 4]
    bits_map = np.array(['00', '01', '10', '11'])
    now = datetime.now()
    accepted = get_last_two_bits(pixels)
    print('accepted:', (datetime.now() - now).total_seconds())
    def AMC_pp(a):
        return bits_map[a & 3]
    now = datetime.now()
    new = AMC_pp(pixels)
    print('new one:', (datetime.now() - now).total_seconds())
    my_output = [bin(x)[-2:].replace('b','0') for x in pixels]
    print(all(my_output == new))
    # print((datetime.now() - now).total_seconds())
    # converter = np.vectorize(lambda x: np.binary_repr(x)[-2:].replace('b', '0'))
    # m2 = converter(pixels)
    

    # a = 0
    # converted = bin(a)[-2:].replace('b', '0')
    # print(converted)

    # sample = np.array([0, 1, 2, 3, 4])
    # bin_sample = [bin(x) for x in sample]
    # print(bin_sample)
    
    # output = [bin(x)[-2:].replace('b','0') for x in sample]
    # print(output)

    # arr_1 = np.random.randint(0, 50, 100)

    # map_list = ['00','01','10','11']
    # def f(x):
    #     return map_list[x % 4]
    # f = np.vectorize(f)
    # output = f(arr_1)
    # my_output = [bin(x)[-2:].replace('b','0') for x in arr_1]

    # print(output == my_output)