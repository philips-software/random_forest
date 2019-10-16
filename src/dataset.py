from dataclasses import dataclass
from typing import Any
from mpyc.runtime import mpc
from src.output import Secret


@dataclass
class ObliviousDataset(Secret):
    rows: [[Any]]
    active_rows: [Any]

    def __init__(self, *rows, active_rows=None):
        if len(rows) == 1:
            self.rows = rows[0]
        else:
            self.rows = rows
        self.active_rows = active_rows

    def column(self, index):
        number_of_columns = len(self.rows[0])
        is_selected = [i == index for i in range(number_of_columns)]
        return mpc.matrix_prod([is_selected], self.rows, True)[0]

    def subset(self, active_rows):
        subset_rows = active_rows
        if self.active_rows:
            subset_rows = mpc.schur_prod(subset_rows, self.active_rows)
        return ObliviousDataset(self.rows, active_rows=subset_rows)

    def __eq__(self, other):
        return list(self.rows) == list(other.rows)

    async def output(self):
        rows = [await mpc.output(row) for row in self.rows]
        if self.active_rows:
            active = await mpc.output(self.active_rows)
        else:
            active = [True] * len(rows)
        return [rows[i] for i in range(len(rows)) if active[i]]
