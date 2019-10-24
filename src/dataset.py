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


class ObliviousDataset(ObliviousArray):

    def column(self, index):
        number_of_columns = len(self.values[0].inputs)
        is_selected = [i == index for i in range(number_of_columns)]
        values = mpc.matrix_prod([is_selected], self.values, True)[0]
        return ObliviousArray(*values, included=self.included)

    def subset(self, included):

        if isinstance(included, ObliviousArray):
            included = list(included.values)

        subset_rows = included
        if self.included:
            subset_rows = mpc.schur_prod(subset_rows, self.included)
        return ObliviousDataset(self.values, included=subset_rows)

    def is_active(self, row_index):
        if self.included == None:
            return s(1)
        return self.included[row_index]

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]
