import numpy as np
from unittest import TestCase


class TestCaseWithArrayEqual(TestCase):
    def assertArrayEqual(self, first: np.ndarray, second: np.ndarray, msg: str = None) -> None:
        if not np.array_equal(first, second):
            self.fail(f'{first} not equal to {second}' + f'{": " + msg if msg else ""}')
