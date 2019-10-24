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
    included: [Any]

    def __init__(self, *rows, included=None):
        if len(rows) == 1 and isinstance(rows[0][0], Sample):
            self.rows = rows[0]
        else:
            self.rows = rows
        self.included = included

    def column(self, index):
        number_of_columns = len(self.rows[0].inputs)
        is_selected = [i == index for i in range(number_of_columns)]
        values = mpc.matrix_prod([is_selected], self.rows, True)[0]
        return ObliviousArray(*values, included=self.included)

    def subset(self, included):

        if isinstance(included, ObliviousArray):
            included = list(included.values)

        subset_rows = included
        if self.included:
            subset_rows = mpc.schur_prod(subset_rows, self.included)
        return ObliviousDataset(self.rows, included=subset_rows)

    def is_active(self, row_index):
        if self.included == None:
            return s(1)
        return self.included[row_index]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    async def output(self):
        rows = [await output(row) for row in self.rows]
        if self.included:
            active = await output(self.included)
        else:
            active = [True] * len(rows)
        return [rows[i] for i in range(len(rows)) if active[i]]
