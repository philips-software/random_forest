from dataclasses import dataclass
from typing import Any
from mpyc.runtime import mpc


@dataclass
class ObliviousArray:
    values: [Any]

    def __init__(self, *values):
        self.values = values

    def __getitem__(self, index):
        is_selected = [i == index for i in range(len(self.values))]
        return mpc.matrix_prod([is_selected], [self.values], True)[0][0]
