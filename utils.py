import numpy as np
from numpy.linalg import norm
from math import sqrt

from typing import List, Union


def similarity(vec_1, vec_2):
    return np.dot(vec_1, vec_2) / (norm(vec_1) * norm(vec_2))


def mean(vectors: List):
    return sum(vectors) / len(vectors)


def l2_norm(vector: List[Union[int, float]]) -> float:
    return sqrt(
        sum([x ** 2 for x in vector])
    )
