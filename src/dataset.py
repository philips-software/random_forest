from dataclasses import dataclass
from typing import Any
from mpyc.runtime import mpc


@dataclass
class ObliviousDataset:
    rows: [[Any]]

    def __init__(self, *matrix_or_rows):
        if len(matrix_or_rows) == 1:
            self.rows = matrix_or_rows[0]
        else:
            self.rows = matrix_or_rows
        self.number_of_rows = len(self.rows)
        self.number_of_columns = len(self.rows[0])

    def column(self, index):
        is_selected = [i == index for i in range(self.number_of_columns)]
        return mpc.matrix_prod([is_selected], self.rows, True)[0]

    def __eq__(self, other):
        return list(self.rows) == list(other.rows)
