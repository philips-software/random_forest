import operator
from dataclasses import dataclass
from typing import Any

from mpyc.runtime import mpc
from mpyc.sectypes import Share

from src.array import ObliviousArray, ObliviousSelection
from src.output import Secret, output
from src.sequence import ObliviousSequence


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
            mpc.vector_add(self.inputs, other.inputs),
            self.outcome + other.outcome
        )

    def __mul__(self, other):
        multiplications = mpc.scalar_mul(other, self.inputs + [self.outcome])
        multiplied_outcome = multiplications.pop()  # mutates multiplications
        return Sample(multiplications, multiplied_outcome)

    async def __output__(self):
        return Sample(await output(self.inputs),
                      await output(self.outcome))


@dataclass(frozen=True)
class __Dataset__(ObliviousSequence):
    number_of_attributes: int

    @property
    def outcomes(self):
        return self.map(lambda sample: sample.outcome)

    def determine_class(self):
        return self.outcomes.sum() * 2 > self.len()

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

    def select(self, *include):
        selection = super().select(*include)
        return ObliviousDatasetSelection(selection, self.number_of_attributes)


class ObliviousDataset(__Dataset__, ObliviousArray):
    def __init__(self, values):
        number_of_attributes = len(values[0]) if len(values) > 0 else 0
        ObliviousArray.__init__(self, values)
        __Dataset__.__init__(self, number_of_attributes)


class ObliviousDatasetSelection(__Dataset__, ObliviousSelection):
    def __init__(self, selection, number_of_attributes):
        ObliviousSelection.__init__(self, selection.values, selection.included)
        __Dataset__.__init__(self, number_of_attributes)
