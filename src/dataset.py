import operator
from dataclasses import dataclass
from typing import Any

from mpyc.runtime import mpc
from mpyc.sectypes import Share

from src.array import ObliviousArray
from src.output import Secret, output


@dataclass
class Sample(Secret):
    inputs: [Any]
    outcome: Any

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index]

    def __add__(self, other):
        return Sample(
            [self.inputs[i] + other.inputs[i] for i in range(len(self))],
            self.outcome + other.outcome
        )

    def __mul__(self, other):
        return Sample(
            [input * other for input in self.inputs],
            self.outcome * other
        )

    async def __output__(self):
        return Sample(await output(self.inputs),
                      await output(self.outcome))


class ObliviousDataset(ObliviousArray):

    def column(self, index):
        if isinstance(index, Share):
            is_selected = mpc.unit_vector(index, self.number_of_attributes)
            return self.map(
                lambda sample:
                ObliviousArray
                .create(sample.inputs)
                .select(is_selected)
                .sum()
            )
        else:
            return self.map(lambda sample: sample.inputs[index])

    @property
    def outcomes(self):
        return self.map(lambda sample: sample.outcome)

    @property
    def number_of_attributes(self):
        return len(self.values[0]) if len(self.values) > 0 else 0
