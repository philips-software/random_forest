from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from mpyc.runtime import mpc
from src.output import Secret, output
from src.array import ObliviousArray

s = mpc.SecInt()


@dataclass
class Sample(Secret):
    inputs: [Any]
    outcome: Any

    def __getitem__(self, index):
        return self.inputs[index]

    async def output(self):
        return Sample(await output(self.inputs),
                      await output(self.outcome))


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
        values = mpc.matrix_prod([is_selected], self.rows, True)[0]
        return ObliviousArray(*values, included=self.active_rows)

    def subset(self, active_rows):

        if isinstance(active_rows, ObliviousArray):
            active_rows = list(active_rows.values)

        subset_rows = active_rows
        if self.active_rows:
            subset_rows = mpc.schur_prod(subset_rows, self.active_rows)
        return ObliviousDataset(self.rows, active_rows=subset_rows)

    def is_active(self, row_index):
        if self.active_rows == None:
            return s(1)
        return self.active_rows[row_index]

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
