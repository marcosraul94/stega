import string
import numpy as np

from core.config import DEFAULT_CONFIG


# text mock related
CHAR_POOL = string.printable + 'áéíóúñ'
# img mock related
LARGE_SIZE = 5000000
BIG_IMG = np.random.randint(0, 2**DEFAULT_CONFIG.num_bits, LARGE_SIZE)
SMALL_IMG = np.arange(2**DEFAULT_CONFIG.num_bits)
# execution time limit
MAX_EXEC_SECONDS = 0.1