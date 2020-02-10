import numpy as np
from core.reader import Reader


phrase = 'hola Cosita es mía'
encoded = phrase.encode()
ints = np.frombuffer(encoded, dtype='uint8')
binaries = [bin(x) for x in ints]
split = np.array([ list(int(value[i:i+2].replace('b', '0'), 2) for i in range(0, 8, 2)) for value in binaries])


last_2_bits = ints & 3
img = np.arange(100).astype('uint8')

img_masked = img & int('11111100', 2)

resized = np.copy(split)
resized.resize(img_masked.shape)
img_with_encoded_bits =  img_masked + resized 

reader = Reader(img_with_encoded_bits)
encoded_rebuilt = reader.read_hidden_bytes()
prhase = encoded_rebuilt.decode()



a = bytearray(encoded)
other_ints = np.array([bin(x) for x in [1, 2, 4]])
print('x[:2]:', other_ints[:2])
print('x[2:]:', other_ints[2:])
for x in other_ints:
    for i in x[2:]:
        print(i in '01')
# print('phrase:', phrase)
# print('encoded:', encoded)
# print('bytearray:', a)
# print('ints bytearray:', np.frombuffer(a, dtype='uint8'))
# print('decoded bytearray:', a.decode())
# print('ints:', ints)
# print('other ints:', other_ints)
# print('last_2_bits:', last_2_bits)
# print('split:', split)
# print('img:', img)
# print('img masked:', img_masked)
# print('resized:', resized)
# print('img with encoded bits:', img_with_encoded_bits)

# print(phrase)