from dataclasses import dataclass
from typing import Any
from mpyc.runtime import mpc
from mpyc.sectypes import Share
from mpyc.random import random_unit_vector
from src.output import Secret, output
from src.array import ObliviousArray
from src.secint import secint


@dataclass
class Sample(Secret):
    inputs: [Any]
    outcome: Any

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index]

    async def __output__(self):
        return Sample(await output(self.inputs),
                      await output(self.outcome))


class ObliviousDataset(ObliviousArray):

    def column(self, index):
        if isinstance(index, Share):
            number_of_columns = len(self.values[0].inputs)
            is_selected = mpc.unit_vector(index, number_of_columns)
            values = mpc.matrix_prod([is_selected], self.values, True)[0]
            return ObliviousArray(values, self.included)
        else:
            values = [row[index] for row in self.values]
            return ObliviousArray(values, self.included)

    @property
    def outcomes(self):
        outs = [sample.outcome for sample in self.values]
        return ObliviousArray(outs, self.included)

    @property
    def number_of_attributes(self):
        return len(self.values[0]) if len(self.values) > 0 else 0

    def random_sample(self):
        selected_row = random_unit_vector(secint, self.len())
        return [self.values[i] * selected_row[i] for i in range(len(selected_row))]
