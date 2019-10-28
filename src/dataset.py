from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from mpyc.runtime import mpc
from mpyc.sectypes import Share
from src.output import Secret, output
from src.array import ObliviousArray

s = mpc.SecInt()


@dataclass
class Sample(Secret):
    inputs: [Any]
    outcome: Any

    def __getitem__(self, index):
        return self.inputs[index]

    async def __output__(self):
        return Sample(await output(self.inputs),
                      await output(self.outcome))


class ObliviousDataset(ObliviousArray):

    def column(self, index):
        if isinstance(index, Share):
            number_of_columns = len(self.values[0].inputs)
            is_selected = [i == index for i in range(number_of_columns)]
            values = mpc.matrix_prod([is_selected], self.values, True)[0]
            return ObliviousArray(*values, included=self.included)
        else:
            values = [row[index] for row in self]
            return ObliviousArray(*values, included=self.included)

    @property
    def outcomes(self):
        outs = [sample.outcome for sample in self.values]
        return ObliviousArray(*outs, included=self.included)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]
