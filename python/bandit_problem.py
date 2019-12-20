import math
import random

import matplotlib.pyplot as plt
from nptyping import Array
import numpy as np
import pandas as pd


class SlotArm():
    def __init__(self, p: float):
        self.p: float = p

    def draw(self):
        if self.p > random.random():
            return 1.0
        else:
            return 0.0

    def initialize(self, n_arms: int) -> None:
        self.n: Array[np.flaot64] = np.zeros(shape=n_arms)
        self.v: Array[np.flaot64] = np.zeros(shape=n_arms)
