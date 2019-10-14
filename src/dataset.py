from dataclasses import dataclass
from typing import Any
from mpyc.runtime import mpc


@dataclass
class ObliviousDataset:
    rows: [[Any]]

    def __init__(self, *rows):
        self.rows = rows
        self.number_of_rows = len(rows)
        self.number_of_columns = len(rows[0])

    def column(self, index):
        is_selected = [i == index for i in range(self.number_of_columns)]
        return mpc.matrix_prod([is_selected], self.rows, True)[0]
