from collections import Sequence
from dataclasses import dataclass
from typing import Any
from mpyc.runtime import mpc
from src.output import Secret, output

s = mpc.SecInt()


@dataclass
class Sample(Secret):
    inputs: [Any]
    output_value: Any

    def __getitem__(self, index):
        return self.inputs[index]

    async def output(self):
        return await output(list(self.inputs))


@dataclass
class ObliviousDataset(Secret):
    rows: [Sample]
    active_rows: [Any]

    def __init__(self, *rows, active_rows=None):
        if len(rows) == 1 and isinstance(rows[0][0], Sample):
            self.rows = rows[0]
        else:
            self.rows = rows
        self.active_rows = active_rows

    def column(self, index):
        number_of_columns = len(self.rows[0].inputs)
        is_selected = [i == index for i in range(number_of_columns)]
        return mpc.matrix_prod([is_selected], self.rows, True)[0]

    def subset(self, active_rows):
        subset_rows = active_rows
        if self.active_rows:
            subset_rows = mpc.schur_prod(subset_rows, self.active_rows)
        return ObliviousDataset(self.rows, active_rows=subset_rows)

    def is_active(self, row_index):
        if self.active_rows == None:
            return s(1)
        return self.active_rows[row_index]

    def __eq__(self, other):
        return list(self.rows) == list(other.rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    async def output(self):
        rows = [await output(row) for row in self.rows]
        if self.active_rows:
            active = await output(self.active_rows)
        else:
            active = [True] * len(rows)
        return [rows[i] for i in range(len(rows)) if active[i]]
