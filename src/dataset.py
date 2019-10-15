from dataclasses import dataclass
from typing import Any
from mpyc.runtime import mpc
from src.output import Secret


@dataclass
class ObliviousDataset(Secret):
    rows: [[Any]]
    active_rows: [Any]

    def __init__(self, *matrix_or_rows):
        if len(matrix_or_rows) == 1:
            self.rows = matrix_or_rows[0]
        else:
            self.rows = matrix_or_rows
        self.number_of_rows = len(self.rows)
        self.number_of_columns = len(self.rows[0])
        self.active_rows = [mpc.SecInt()(1) for _ in self.rows]

    def column(self, index):
        is_selected = [i == index for i in range(self.number_of_columns)]
        return mpc.matrix_prod([is_selected], self.rows, True)[0]

    def select_rows(self, active_rows):
        self.active_rows = mpc.schur_prod(self.active_rows, active_rows)

    def __eq__(self, other):
        return list(self.rows) == list(other.rows)

    async def output(self):
        rows = [await mpc.output(row) for row in self.rows]
        active = await mpc.output(self.active_rows)
        return [rows[i] for i in range(len(rows)) if active[i]]
